# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DDP/FSDP wrapping with mixed precision managed by the parallel strategy."""

import logging

import torch
import torch._dynamo
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP

from .activation_checkpointing import apply_activation_checkpointing
from .context import DistributedContext
from .utils import (
    filter_supported_kwargs,
    get_module_names_by_dtype,
    is_container_module,
    is_rank_zero,
    module_param_dtypes,
    module_param_numel,
    module_params,
    parse_optional_int_list,
)

logger = logging.getLogger(__name__)


def _unwrap_checkpoint_module(module: nn.Module) -> nn.Module:
    """Return the original module registered by a checkpoint wrapper."""
    return getattr(module, "_checkpoint_wrapped_module", module)


def _is_fsdp_boundary(module: nn.Module) -> bool:
    """Return whether a module can serve as an FSDP hook boundary.

    FSDP installs communication hooks on ``module.forward``, so a boundary
    must implement ``forward`` and cannot be a traversal-only container.
    """
    return (
        not is_container_module(module)
        and module.__class__.forward is not nn.Module.forward
    )


def wrap_model(model: nn.Module, training_args, ctx: DistributedContext) -> nn.Module:
    """Wrap model with DDP or FSDP based on CLI training_args; mixed precision included."""
    # Lazy import: loongforge.embodied.train imports back into
    # loongforge.embodied.distributed.parallel (trainer_builder -> groot_trainer ->
    # wrap_model), so a module-level import here creates a circular import.
    from loongforge.embodied.train.utils.utils import resolve_dtype

    dtype = resolve_dtype(training_args.dtype)
    apply_activation_checkpointing(
        model,
        training_args.activation_checkpoint_module_patterns,
        training_args.activation_checkpoint_skip_modules,
    )

    if not ctx.is_distributed:
        if len(get_module_names_by_dtype(model, trainable_only=False)) > 1:
            return model.to(device=ctx.device)
        return model.to(dtype=dtype, device=ctx.device)

    strategy = training_args.distributed_strategy
    if strategy == "fsdp":
        return _wrap_fsdp(model, training_args, ctx, dtype)
    else:
        return _wrap_ddp(model, training_args, ctx, dtype)


def _wrap_ddp(model: nn.Module, training_args, ctx: DistributedContext, dtype: torch.dtype) -> nn.Module:
    """Wrap model with DistributedDataParallel."""
    all_modules_by_dtype = get_module_names_by_dtype(model, trainable_only=False)
    if len(all_modules_by_dtype) > 1:
        model = model.to(device=ctx.device)
    else:
        model = model.to(dtype=dtype, device=ctx.device)

    torch._dynamo.config.optimize_ddp = training_args.dynamo_optimize_ddp

    ddp_kwargs = {
        "broadcast_buffers": training_args.ddp_broadcast_buffers,
        "init_sync": training_args.ddp_init_sync,
        "bucket_cap_mb": training_args.ddp_bucket_cap_mb,
        "find_unused_parameters": training_args.ddp_find_unused_parameters,
        "gradient_as_bucket_view": training_args.ddp_gradient_as_bucket_view,
        "static_graph": training_args.ddp_static_graph,
        "skip_all_reduce_unused_params": training_args.ddp_skip_all_reduce_unused_params,
        "bucket_cap_mb_list": parse_optional_int_list(training_args.ddp_bucket_cap_mb_list),
        "batched_grad_copy": training_args.ddp_batched_grad_copy,
    }

    return DDP(model, **filter_supported_kwargs(DDP, ddp_kwargs))


def _wrap_fsdp(
    model: nn.Module,
    training_args,
    ctx: DistributedContext,
    dtype: torch.dtype,
) -> nn.Module:
    """Apply FSDP2 with dtype-safe, bottom-up wrapping.

    Wrapping has two important constraints:
    1. One FSDP communication group may only contain parameters with the same
       original dtype. Mixed-dtype candidates must be split before calling
       ``fully_shard`` on the candidate itself.
    2. ``fully_shard`` installs all-gather/reshard hooks on the wrapped
       module's ``forward``. Some registered modules are only structural
       containers or helper parameter owners and are never called directly by
       ``model.forward``; those modules are unsafe hook boundaries, so the
       planner must descend to callable children instead.

    Without configured module classes, the generic planner wraps from inner
    modules to outer modules so later grouping decisions can exclude parameters
    already assigned to child groups. Its stages are:
    1. Auto-wrap repeated, parameter-heavy callable modules as likely
       layer/block boundaries.
    2. Wrap remaining large, forward-executed leaf modules by dtype, avoiding
       parameter holders whose FSDP hooks would never run.
    3. Wrap the root last as the catch-all group for any remaining parameters
       and to expose a top-level FSDPModule to trainer/checkpoint code.

    For every candidate, child modules are recursively wrapped or descended
    into until the candidate's remaining parameters are dtype-uniform.
    Composable FSDP excludes already sharded child groups automatically; the
    planner tracks parameter ids only for its own grouping decisions.
    """
    from loongforge.embodied.train.training_args import resolve_fsdp_dtype

    if ctx.device.type == "cuda":
        torch.cuda.set_device(ctx.device)

    module_class_names = {
        name.strip()
        for name in (training_args.fsdp_wrap_modules or "").split(",")
        if name.strip()
    }
    configured_units = (
        _resolve_fsdp_units(model, module_class_names)
        if module_class_names
        else []
    )
    modules_by_dtype = get_module_names_by_dtype(model, trainable_only=False)
    authored_mixed_dtype = len(modules_by_dtype) > 1
    if training_args.fsdp_original_param_dtype is None:
        original_param_dtype = None if authored_mixed_dtype else dtype
    else:
        original_param_dtype = resolve_fsdp_dtype(
            training_args.fsdp_original_param_dtype
        )
    if training_args.fsdp_unsharded_param_dtype is None:
        unsharded_param_dtype = None if original_param_dtype is None else dtype
    else:
        unsharded_param_dtype = resolve_fsdp_dtype(
            training_args.fsdp_unsharded_param_dtype
        )
    reduce_dtype = resolve_fsdp_dtype(training_args.fsdp_reduce_dtype)
    if original_param_dtype is not None:
        model.to(dtype=original_param_dtype)

    scalar_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.ndim == 0
    ]
    trainable_scalar_names = [
        name for name, param in scalar_params if param.requires_grad
    ]
    if trainable_scalar_names:
        raise ValueError(
            "FSDP2 cannot ignore trainable scalar parameters because their "
            "gradients would not be reduced: "
            + ", ".join(trainable_scalar_names[:8])
        )
    ignored_params = {param for _, param in scalar_params}
    if ignored_params:
        with torch.no_grad():
            for param in ignored_params:
                if param.device != ctx.device:
                    param.data = param.to(device=ctx.device)
        logger.info(
            "FSDP2 ignores %d frozen scalar params unsupported by fully_shard: %s",
            len(ignored_params),
            ", ".join(name for name, _ in scalar_params[:8]),
        )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=unsharded_param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=training_args.fsdp_cast_forward_inputs,
    )
    logger.info(
        "FSDP2 runtime: training_dtype=%s original_param_dtype=%s "
        "unsharded_param_dtype=%s reduce_dtype=%s cast_forward_inputs=%s",
        dtype,
        original_param_dtype,
        unsharded_param_dtype,
        reduce_dtype,
        training_args.fsdp_cast_forward_inputs,
    )

    dp_mesh = _build_fsdp_device_mesh(training_args, ctx)

    fsdp_kwargs = {
        "mesh": dp_mesh,
    }

    # The planner mutates ``model`` in-place by calling ``fully_shard`` on each
    # selected module. It tracks wrapped parameter ids so later planner stages
    # only inspect parameters not already assigned to an inner group.
    planner = _FSDPWrapPlanner(
        model=model,
        training_args=training_args,
        fsdp_kwargs=fsdp_kwargs,
        mp_policy=mp_policy,
        ignored_params=ignored_params,
    )

    # Configured classes supply exact inner FSDP units; empty configuration
    # continues through the generic planner.
    if configured_units:
        prefetch_module_runs = _resolve_ordered_prefetch_module_runs(
            model,
            configured_units,
        )
        wrapped_unit_count = sum(
            planner._safe_fully_shard(module, name=module_name)
            for module_name, module in configured_units
        )
        root_group_created = planner._safe_fully_shard(model, name="<root>")
        for prefetch_modules in prefetch_module_runs:
            _configure_fsdp_prefetch(
                prefetch_modules,
                training_args.fsdp_forward_prefetch_distance,
                training_args.fsdp_backward_prefetch_distance,
            )
        prefetch_unit_count = sum(len(run) for run in prefetch_module_runs)
        logger.info(
            "FSDP wrapped %d module groups "
            "(configured_units=%d, prefetch_units=%d, root=%s).",
            planner.num_wrapped_groups,
            wrapped_unit_count,
            prefetch_unit_count,
            root_group_created,
        )
        return model

    # Order matters: inner groups must be created before parent/root groups so
    # later planner stages only consider residual parameters.
    repeated_group_count = planner.wrap_repeated_layer_modules()
    leftover_group_count = planner.wrap_leftover_leaf_modules_by_dtype()
    root_group_created = planner.wrap_root()
    logger.info(
        "FSDP wrapped %d module groups "
        "(repeated=%d, leftover=%d, root=%s).",
        planner.num_wrapped_groups,
        repeated_group_count,
        leftover_group_count,
        root_group_created,
    )

    return model


def _resolve_fsdp_units(
    model: nn.Module,
    class_names: set[str],
) -> list[tuple[str, nn.Module]]:
    """Resolve explicit class-name units, including checkpoint wrappers."""
    # CheckpointWrapper registers the original module as a child. Match its
    # class on the outer wrapper so FSDP hooks surround the checkpointed call.
    checkpoint_wrapped_module_ids = {
        id(original_module)
        for module in model.modules()
        if (original_module := _unwrap_checkpoint_module(module)) is not module
    }
    selected_units = {}
    matched_class_names = set()
    for module_key, module in model.named_modules():
        if not module_key or id(module) in checkpoint_wrapped_module_ids:
            continue
        original_module = _unwrap_checkpoint_module(module)
        class_name = original_module.__class__.__name__
        if class_name not in class_names:
            continue
        if not _is_fsdp_boundary(module):
            raise ValueError(
                f"FSDP unit {module_key!r} is not a callable module boundary"
            )
        if next(module.parameters(recurse=True), None) is None:
            raise ValueError(f"FSDP unit {module_key!r} has no parameters")
        selected_units[module_key] = module
        matched_class_names.add(class_name)

    unmatched_class_names = class_names.difference(matched_class_names)
    if unmatched_class_names:
        raise ValueError(
            "FSDP wrap module classes matched no callable boundaries: "
            + ", ".join(sorted(unmatched_class_names))
        )
    return sorted(
        selected_units.items(),
        key=lambda item: item[0].count("."),
        reverse=True,
    )


def _resolve_ordered_prefetch_module_runs(
    model: nn.Module,
    units: list[tuple[str, nn.Module]],
) -> list[list[nn.Module]]:
    """Find fully selected same-class children under ordered containers.

    This supports straight-line execution only: each child must run once in
    registration order, without reordering, skipping, repetition, or branches.
    """
    selected_modules = dict(units)
    prefetch_runs = []
    for parent_name, parent_module in model.named_modules():
        if not isinstance(parent_module, (nn.ModuleList, nn.Sequential)):
            continue
        child_modules = [
            selected_modules.get(
                f"{parent_name}.{child_name}" if parent_name else child_name
            )
            for child_name, _ in parent_module.named_children()
        ]
        if len(child_modules) <= 1 or any(module is None for module in child_modules):
            continue
        module_classes = {
            _unwrap_checkpoint_module(module).__class__
            for module in child_modules
        }
        if len(module_classes) == 1:
            prefetch_runs.append(child_modules)

    logger.info(
        "FSDP2 inferred %d registration-ordered prefetch runs",
        len(prefetch_runs),
    )
    return prefetch_runs


def _configure_fsdp_prefetch(
    fsdp_modules: list[nn.Module],
    forward_distance: int,
    backward_distance: int,
) -> None:
    """Configure FSDP2 prefetch edges from an ordered module sequence."""
    if forward_distance <= 0 and backward_distance <= 0:
        return

    fsdp_modules = [
        module
        for module in fsdp_modules
        if isinstance(module, FSDPModule)
    ]
    if len(fsdp_modules) <= 1:
        logger.info(
            "FSDP2 prefetch requested but only %d FSDP modules found",
            len(fsdp_modules),
        )
        return

    forward_prefetch_supported = hasattr(
        fsdp_modules[0], "set_modules_to_forward_prefetch"
    )
    backward_prefetch_supported = hasattr(
        fsdp_modules[0], "set_modules_to_backward_prefetch"
    )
    unsupported_directions = []
    if forward_distance > 0 and not forward_prefetch_supported:
        unsupported_directions.append("forward")
    if backward_distance > 0 and not backward_prefetch_supported:
        unsupported_directions.append("backward")
    if unsupported_directions:
        logger.warning(
            "FSDP %s prefetch requested but this PyTorch FSDP2 version has no "
            "corresponding prefetch API",
            "/".join(unsupported_directions),
        )

    forward_edges = _set_fsdp_prefetch_edges(
        fsdp_modules,
        forward_distance if forward_prefetch_supported else 0,
        forward=True,
    )
    backward_edges = _set_fsdp_prefetch_edges(
        fsdp_modules,
        backward_distance if backward_prefetch_supported else 0,
        forward=False,
    )

    logger.info(
        "FSDP2 prefetch configured: modules=%d forward_distance=%d "
        "backward_distance=%d forward_edges=%d backward_edges=%d",
        len(fsdp_modules),
        forward_distance,
        backward_distance,
        forward_edges,
        backward_edges,
    )


def _set_fsdp_prefetch_edges(
    fsdp_modules: list[nn.Module],
    distance: int,
    *,
    forward: bool,
) -> int:
    """Configure one direction of FSDP2 prefetch and return its edge count."""
    if distance <= 0:
        return 0

    edge_count = 0
    for index, current_module in enumerate(fsdp_modules):
        if forward:
            targets = fsdp_modules[index + 1 : index + 1 + distance]
            setter = current_module.set_modules_to_forward_prefetch
        else:
            start = max(0, index - distance)
            targets = list(reversed(fsdp_modules[start:index]))
            setter = current_module.set_modules_to_backward_prefetch
        if not targets:
            continue
        setter(targets)
        edge_count += len(targets)
    return edge_count


def _build_fsdp_device_mesh(training_args, ctx: DistributedContext):
    """Build the FSDP/HSDP device mesh used by fully_shard.

    FSDP uses a 1D mesh and shards parameters across all data-parallel ranks.
    HSDP uses a 2D mesh: dim 0 is replicated and dim 1 is sharded. Passing
    ``--hsdp-shard-size`` enables HSDP and sets the sharding dimension size.
    """
    shard_size = training_args.hsdp_shard_size
    if shard_size is None:
        return init_device_mesh(
            "cuda",
            (ctx.world_size,),
            mesh_dim_names=("dp",),
        )

    if shard_size <= 0:
        raise ValueError(f"HSDP shard size must be positive, got {shard_size}.")
    if ctx.world_size % shard_size != 0:
        raise ValueError(
            "HSDP requires world_size to be divisible by hsdp_shard_size, "
            f"got world_size={ctx.world_size}, hsdp_shard_size={shard_size}."
        )

    replica_size = ctx.world_size // shard_size
    if replica_size <= 1:
        logger.warning(
            "--hsdp-shard-size is set with one replica group; this is equivalent "
            "to FSDP over the HSDP shard group."
        )

    if is_rank_zero():
        logger.info(
            "Using HSDP 2D device mesh: replica=%d, shard=%d.",
            replica_size,
            shard_size,
        )

    return init_device_mesh(
        "cuda",
        (replica_size, shard_size),
        mesh_dim_names=("replica", "shard"),
    )


class _FSDPWrapPlanner:
    """Build and apply dtype-valid FSDP2 groups for a module tree.

    The planner is deliberately generic: model-specific exclusions enter
    through CLI values instead of hard-coded Python constants.
    """

    def __init__(
        self,
        model: nn.Module,
        training_args,
        fsdp_kwargs: dict,
        mp_policy: MixedPrecisionPolicy,
        ignored_params: set[nn.Parameter] | None = None,
    ):
        """Initialize planner state, precision policies, and wrap thresholds."""
        self.model = model
        self.fsdp_kwargs = fsdp_kwargs
        self.mp_policy = mp_policy

        # The repeated-layer and leftover stages use separate thresholds:
        # - repeated_min_num_params controls layer/block auto wrapping;
        # - leftover_min_num_params controls large leaf cleanup wrapping.
        # Keeping the leftover threshold configurable avoids wrapping every tiny
        # Linear/Norm leaf, which can increase communication and hook overhead.
        self.repeated_min_num_params = int(training_args.fsdp_min_num_params)
        self.leftover_min_num_params = int(training_args.fsdp_leftover_min_num_params)

        self.fsdp_reshard_default = training_args.fsdp_reshard_default
        self.fsdp_reshard_root = training_args.fsdp_reshard_root
        self.fsdp_reshard_module_overrides = (
            training_args.fsdp_reshard_module_overrides or {}
        )
        # Classes listed here are not wrapped as a single boundary. The planner
        # descends into their children instead. This is useful for modules whose
        # parameters are read by custom code outside that module's forward hooks.
        extra_no_wrap = training_args.fsdp_no_wrap_modules
        self.no_wrap_module_classes = {
            name.strip() for name in extra_no_wrap.split(",") if name.strip()
        } if extra_no_wrap else set()

        # FSDP2 excludes earlier child groups automatically. Parameter ids are
        # tracked only for planner decisions; explicit ignores are frozen scalars.
        self.ignored_params = set(ignored_params or set())
        self.wrapped_module_ids = set()
        self.wrapped_param_ids = {id(param) for param in self.ignored_params}
        self.wrapped_group_count = 0

    @property
    def num_wrapped_groups(self) -> int:
        """Number of FSDP groups created by fully_shard."""
        return self.wrapped_group_count

    def wrap_repeated_layer_modules(self) -> int:
        """Wrap repeated modules as likely layer boundaries.

        The heuristic targets module classes that appear multiple times and own
        enough parameters collectively to justify an FSDP group. Small sibling
        layers can be grouped together until the group reaches the configured
        threshold. This matches common model structures such as transformer
        decoder layers or vision encoder blocks.
        """
        # Count duplicate references using remove_duplicate=False. Shared module
        # instances are skipped below because sharding the same object through
        # multiple names would make the group selection ambiguous.
        module_occurrences = {}
        for _, module in self.model.named_modules(remove_duplicate=False):
            module_occurrences[id(module)] = module_occurrences.get(id(module), 0) + 1

        # Collect possible repeated layer boundaries. This pass only filters out
        # modules that are clearly unsafe or out of scope; it does not apply the
        # size threshold yet because adjacent sibling layers may be grouped into
        # one FSDP communication group below.
        #
        # Keep modules that:
        # - are not the root, which ``wrap_root()`` handles last;
        # - are not structural containers or user-declared no-wrap classes;
        # - have children, leaving leaf parameter owners for the leftover stage;
        # - are not shared module instances appearing under multiple names;
        # - still own unwrapped parameters.
        repeated_boundary_candidates = [
            (name, module)
            for name, module in self._named_modules()
            if (
                name
                and id(module) not in self.wrapped_module_ids
                and _is_fsdp_boundary(module)
                and not self._is_no_wrap_module(module)
                and any(True for _ in module.children())
                and module_occurrences.get(id(module), 0) == 1
                and module_param_numel(module, excluded_param_ids=self.wrapped_param_ids) > 0
            )
        ]

        # A single large helper class is not necessarily a repeated layer. Count
        # candidate classes and keep only classes that appear multiple times,
        # e.g. ``SiglipEncoderLayer: 27`` or ``GemmaMLP: 36`` are layer-like,
        # while one-off classes such as ``PI05Pytorch: 1`` are not.
        class_counts = {}
        for _, module in repeated_boundary_candidates:
            class_name = module.__class__.__name__
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        repeated_layer_candidates = [
            (name, module)
            for name, module in repeated_boundary_candidates
            if class_counts.get(module.__class__.__name__, 0) > 1
        ]

        # If nested candidates both satisfy the repeated-layer heuristic, keep
        # only the outermost candidate for this stage. The dtype split logic can
        # still descend later if the outer candidate cannot be wrapped directly.
        candidate_names = {name for name, _ in repeated_layer_candidates}
        repeated_layer_candidates = [
            (name, module)
            for name, module in repeated_layer_candidates
            if not any(
                name.startswith(parent_name + ".")
                for parent_name in candidate_names
                if parent_name != name
            )
        ]

        # Build contiguous sibling runs before applying the size threshold. FSDP
        # list groups should combine only modules that share the same direct
        # parent and appear next to each other with the same class, e.g.
        # ``layers.0, layers.1, layers.2``.
        candidate_by_name = dict(repeated_layer_candidates)
        repeated_layer_groups = []
        for parent_name, parent_module in self._named_modules():
            # ``sibling_candidate_run`` is one contiguous same-class candidate
            # span under this parent. ``_chunk_repeated_layer_run`` later splits
            # it into one or more threshold-sized repeated layer groups.
            sibling_candidate_run = []
            run_class_name = None
            for child_name, child in parent_module.named_children():
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                child_candidate = candidate_by_name.get(full_name)
                if child_candidate is None:
                    # A non-candidate child breaks the contiguous run.
                    repeated_layer_groups.extend(
                        self._chunk_repeated_layer_run(sibling_candidate_run)
                    )
                    sibling_candidate_run = []
                    run_class_name = None
                    continue

                class_name = child_candidate.__class__.__name__
                if sibling_candidate_run and class_name != run_class_name:
                    # Different repeated classes should form separate groups.
                    repeated_layer_groups.extend(
                        self._chunk_repeated_layer_run(sibling_candidate_run)
                    )
                    sibling_candidate_run = []
                sibling_candidate_run.append(child_candidate)
                run_class_name = class_name

            # Flush the final run for this parent.
            repeated_layer_groups.extend(
                self._chunk_repeated_layer_run(sibling_candidate_run)
            )

        repeated_group_count = 0
        for group in repeated_layer_groups:
            repeated_group_count += self._wrap_repeated_layer_group(group)
        logger.info("FSDP wrapping auto-selected repeated module classes.")
        return repeated_group_count

    def _chunk_repeated_layer_run(self, modules: list[nn.Module]) -> list[list[nn.Module]]:
        """Group adjacent repeated layers until each group reaches the threshold."""
        if not modules:
            return []

        groups = []
        current_group = []
        current_numel = 0
        for module in modules:
            numel = module_param_numel(module, excluded_param_ids=self.wrapped_param_ids)
            if numel <= 0:
                continue
            current_group.append(module)
            current_numel += numel
            if current_numel >= self.repeated_min_num_params:
                groups.append(current_group)
                current_group = []
                current_numel = 0

        if current_group:
            if groups:
                groups[-1].extend(current_group)
            elif current_numel >= self.repeated_min_num_params:
                groups.append(current_group)
        return groups

    def _wrap_repeated_layer_group(self, modules: list[nn.Module]) -> int:
        """Wrap one repeated-layer group and return created FSDP group count.

        The input group has already been selected as adjacent repeated layers.
        This method performs the final safety pass before calling
        ``fully_shard``:
        1. Drop modules already assigned to an inner/earlier FSDP group.
        2. Recursively split mixed-dtype children out of each module.
        3. Scan the remaining modules in order and collect contiguous modules
           with the same remaining dtype.
        4. Wrap each same-dtype run as one FSDP communication group.
        """
        modules = [module for module in modules if id(module) not in self.wrapped_module_ids]
        if not modules:
            return 0

        # Each individual layer must expose at most one remaining dtype before
        # it can participate in a list-based FSDP group.
        for module in modules:
            self._make_candidate_params_uniform(module)

        created_group_count = 0
        same_dtype_group = []
        same_dtype = None
        for module in modules:
            dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
            if not dtypes:
                continue
            if len(dtypes) > 1:
                # Flush the current same-dtype run, then fall back to the
                # generic candidate wrapper so it can continue recursive dtype
                # splitting for this difficult module.
                created_group_count += self._wrap_same_dtype_module_group(
                    same_dtype_group
                )
                same_dtype_group = []
                same_dtype = None
                created_group_count += int(self._wrap_candidate(module))
                continue

            dtype = next(iter(dtypes))
            if same_dtype_group and dtype != same_dtype:
                # A dtype change starts a new FSDP communication group.
                created_group_count += self._wrap_same_dtype_module_group(
                    same_dtype_group
                )
                same_dtype_group = []
            same_dtype_group.append(module)
            same_dtype = dtype

        created_group_count += self._wrap_same_dtype_module_group(same_dtype_group)
        return created_group_count

    def _wrap_same_dtype_module_group(
        self,
        modules: list[nn.Module],
    ) -> int:
        """Wrap one non-empty same-dtype module run as an FSDP group."""
        if not modules:
            return 0
        target = modules[0] if len(modules) == 1 else modules
        return int(self._safe_fully_shard(target))

    def wrap_leftover_leaf_modules_by_dtype(self) -> int:
        """Wrap remaining leaf parameter owners that exceed the leftover threshold.

        This cleanup stage handles large callable leaves missed by the explicit
        and repeated-layer stages. It ignores structural parameter holders and
        modules with children because those need a parent/root execution boundary
        or the recursive dtype splitter.
        """
        leftover_group_count = 0
        for name, module in self._named_modules():
            if not name:
                continue
            if id(module) in self.wrapped_module_ids:
                continue
            if not _is_fsdp_boundary(module):
                continue
            if any(True for _ in module.children()):
                continue
            if module_param_numel(module) < self.leftover_min_num_params:
                continue
            if self._wrap_candidate(module):
                leftover_group_count += 1
        return leftover_group_count

    def wrap_root(self) -> bool:
        """Wrap the root module last.

        Root wrapping is a catch-all for residual parameters not assigned to
        inner groups and exposes a top-level FSDPModule to trainer/checkpoint
        code. Composable FSDP excludes previously sharded child groups.
        """
        return self._wrap_candidate(self.model, force=True)

    def _wrap_candidate(
        self,
        module: nn.Module,
        force: bool = False,
    ) -> bool:
        """Wrap module after first making its remaining parameters dtype-uniform.

        FSDP2 requires each flattened group to have a single original parameter
        dtype. If a candidate still contains multiple dtypes after ignoring
        already wrapped inner groups, the planner recursively wraps children
        until the candidate's remaining parameters are dtype-uniform.
        """
        # Structural parameter holders are traversal nodes, not execution
        # boundaries. Their children may still be valid FSDP units.
        if id(module) in self.wrapped_module_ids:
            return False
        if module is not self.model and not _is_fsdp_boundary(module):
            return self._wrap_child_boundaries(module)

        # A no-wrap class is not a dead end. It means "do not use this module as
        # the FSDP hook boundary"; its children may still be valid boundaries.
        if self._is_no_wrap_module(module):
            return self._wrap_child_boundaries(module)

        self._make_candidate_params_uniform(module)
        dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
        if not dtypes:
            if force:
                return self._safe_fully_shard(module)
            return False
        if len(dtypes) > 1:
            raise ValueError(
                f"Unable to derive a uniform-dtype FSDP group for "
                f"{module.__class__.__name__}: {dtypes}."
            )
        return self._safe_fully_shard(module)

    def _named_modules(self) -> list[tuple[str, nn.Module]]:
        """Return deduplicated modules in traversal order."""
        return list(self.model.named_modules(remove_duplicate=True))

    def _is_no_wrap_module(self, module: nn.Module) -> bool:
        """Return whether ``module`` is excluded as an FSDP boundary."""
        return module is not self.model and module.__class__.__name__ in self.no_wrap_module_classes

    def _make_candidate_params_uniform(self, module: nn.Module) -> None:
        """Recursively isolate minority dtype children before wrapping module.

        The dominant dtype, measured by parameter numel, stays in the current
        candidate. Children that only contain other dtypes are wrapped or
        descended into first. After those child groups are recorded as wrapped,
        the current candidate should expose at most one remaining dtype.
        """
        dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
        if len(dtypes) <= 1:
            return

        dtype_numel = {}
        for param in module_params(module, excluded_param_ids=self.wrapped_param_ids):
            dtype_numel[param.dtype] = dtype_numel.get(param.dtype, 0) + param.numel()
        target_dtype = max(dtype_numel, key=dtype_numel.get)

        # First pass: isolate children that do not contain the dominant dtype,
        # and recursively split children that still contain multiple dtypes.
        for child in module.children():
            if id(child) in self.wrapped_module_ids:
                continue
            child_dtypes = module_param_dtypes(child, excluded_param_ids=self.wrapped_param_ids)
            if not child_dtypes:
                continue
            if target_dtype not in child_dtypes:
                self._wrap_valid_boundary_or_children(child)
            elif len(child_dtypes) > 1:
                self._make_candidate_params_uniform(child)

        dtypes = module_param_dtypes(module, excluded_param_ids=self.wrapped_param_ids)
        if len(dtypes) <= 1:
            return

        # Second pass: if mixed dtypes remain, wrap any child whose remaining
        # dtype set differs from the dominant dtype. This handles cases where a
        # child contains the dominant dtype plus another dtype after recursion.
        for child in module.children():
            if id(child) in self.wrapped_module_ids:
                continue
            child_dtypes = module_param_dtypes(child, excluded_param_ids=self.wrapped_param_ids)
            if child_dtypes and child_dtypes != {target_dtype}:
                self._wrap_valid_boundary_or_children(child)

    def _wrap_valid_boundary_or_children(self, module: nn.Module) -> bool:
        """Wrap a valid FSDP boundary, or keep looking below a structural module."""
        if not _is_fsdp_boundary(module):
            return self._wrap_child_boundaries(module)
        return self._wrap_candidate(module)

    def _wrap_child_boundaries(self, module: nn.Module) -> bool:
        """Try to wrap valid FSDP boundaries under this module's direct children."""
        wrapped_any = False
        for child in module.children():
            wrapped_any = self._wrap_valid_boundary_or_children(child) or wrapped_any
        return wrapped_any

    def _safe_fully_shard(
        self,
        module_or_modules: nn.Module | list[nn.Module],
        *,
        name: str | None = None,
    ) -> bool:
        """Shard one dtype-safe module group and record its parameter ownership."""
        modules = self._as_module_group(module_or_modules)
        if not modules or any(id(module) in self.wrapped_module_ids for module in modules):
            return False

        params_before = self._module_group_params(modules)
        dtypes = {param.dtype for param in params_before}
        if len(dtypes) > 1:
            group_label = name or self._module_group_label(modules)
            raise ValueError(
                f"FSDP cannot wrap mixed original dtypes {dtypes} in "
                f"{group_label}."
            )

        fully_shard_kwargs = dict(self.fsdp_kwargs)
        reshard_after_forward = self._reshard_after_forward_for_group(modules)
        if reshard_after_forward is not None:
            fully_shard_kwargs["reshard_after_forward"] = reshard_after_forward

        fully_shard(
            module_or_modules,
            mp_policy=self.mp_policy,
            ignored_params=self.ignored_params,
            **fully_shard_kwargs,
        )
        self._mark_wrapped(modules, params_before)
        return True

    def _as_module_group(self, module_or_modules: nn.Module | list[nn.Module]) -> list[nn.Module]:
        """Normalize one module or a module list into a list."""
        return [module_or_modules] if isinstance(module_or_modules, nn.Module) else list(module_or_modules)

    def _module_group_params(self, modules: list[nn.Module]) -> list[nn.Parameter]:
        """Return deduplicated unwrapped parameters owned by ``modules``."""
        all_params = [
            p
            for module in modules
            for p in module_params(module, excluded_param_ids=self.wrapped_param_ids)
        ]
        # Deduplicate across modules while preserving order.
        return list({id(p): p for p in all_params}.values())

    def _module_group_label(self, modules: list[nn.Module]) -> str:
        """Build a descriptive module-group label for logs and errors."""
        if len(modules) == 1:
            return modules[0].__class__.__name__
        class_names = {module.__class__.__name__ for module in modules}
        class_label = next(iter(class_names)) if len(class_names) == 1 else "mixed classes"
        return f"{len(modules)} modules ({class_label})"

    def _reshard_after_forward_for_group(self, modules: list[nn.Module]) -> bool | int | None:
        """Resolve one shared reshard policy for a module group."""
        values = [self._reshard_after_forward_for(module) for module in modules]
        first_value = values[0]
        if any(value != first_value for value in values):
            raise ValueError(
                "Cannot create one FSDP group with different "
                f"reshard_after_forward settings: {values}."
            )
        return first_value

    def _reshard_after_forward_for(self, module: nn.Module) -> bool | int | None:
        """Resolve the configured reshard policy for one module."""
        original_module = _unwrap_checkpoint_module(module)
        class_name = original_module.__class__.__name__
        if class_name in self.fsdp_reshard_module_overrides:
            return self.fsdp_reshard_module_overrides[class_name]
        if module is self.model:
            return self.fsdp_reshard_root
        return self.fsdp_reshard_default

    def _mark_wrapped(
        self,
        modules: list[nn.Module],
        params_before: list[nn.Parameter],
    ) -> None:
        """Record module and parameter ownership after a successful fully_shard."""
        self.wrapped_group_count += 1

        for module in modules:
            self.wrapped_module_ids.add(id(module))
        for param in params_before:
            self.wrapped_param_ids.add(id(param))

        # FSDP replaces parameters during wrapping, so record current ids too.
        for module in modules:
            for param in module.parameters(recurse=True):
                self.wrapped_param_ids.add(id(param))
