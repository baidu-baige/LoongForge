# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Runtime monkey-patches for DP balance adaptation.

Registers and applies patches to PyTorch's _pin_memory_loop and Megatron's
RerunDataIterator to enable cross-DP data reordering during training.
"""

import importlib
import sys
import types
from importlib.machinery import ModuleSpec


def dummy_function_wrapper(func_name):
    """
    Create a placeholder function that raises a RuntimeError when invoked.

    This is used when a target function does not exist but `create_dummy=True`
    is specified. It allows the patching process to continue without failing
    at import time.

    Args:
        func_name (str): Name of the missing function.

    Returns:
        function: A callable that raises RuntimeError when called.
    """

    def dummy_function(*args, **kwargs):
        raise RuntimeError(f"function {func_name} no exist")

    return dummy_function


def parse_path(module_path, function_name, create_dummy):
    """
    Dynamically import a module and optionally resolve a function within it.

    This function walks through the module path progressively, importing
    submodules one by one. If a module does not exist and `create_dummy`
    is True, a virtual module will be created and registered in sys.modules.

    Args:
        module_path (str): Full module path (e.g., "package.sub.module").
        function_name (str or None): Name of the function inside the module.
        create_dummy (bool): Whether to create missing modules/functions.

    Returns:
        tuple:
            module (module): Imported or dynamically created module.
            function (callable or None): Resolved function object if specified.

    Raises:
        ModuleNotFoundError: If module is missing and create_dummy=False.
        RuntimeError: If function is missing and create_dummy=False.
    """
    modules = module_path.split(".")

    for i in range(1, len(modules) + 1):
        parent = ".".join(modules[: i - 1])
        path = ".".join(modules[:i])

        try:
            importlib.import_module(path)
        except ModuleNotFoundError as e:
            if not parent or not hasattr(
                importlib.import_module(parent), modules[i - 1]
            ):
                if not create_dummy:
                    raise ModuleNotFoundError(e) from e

                # create dummy module
                sys.modules[path] = types.ModuleType(path)
                sys.modules[path].__file__ = "dummy_module.py"
                sys.modules[path].__spec__ = ModuleSpec(path, None)

                if parent:
                    setattr(
                        importlib.import_module(parent),
                        modules[i - 1],
                        sys.modules[path],
                    )
            else:
                module = getattr(importlib.import_module(parent), modules[i - 1])
                if hasattr(module, function_name):
                    return module, getattr(module, function_name)
                elif create_dummy:
                    return module, dummy_function_wrapper(function_name)
                else:
                    raise RuntimeError(f"no exist {function_name} of {module}")

    if function_name is not None and not hasattr(
        sys.modules[module_path], function_name
    ):
        setattr(sys.modules[module_path], function_name, None)

    return (
        sys.modules[module_path],
        (
            getattr(sys.modules[module_path], function_name)
            if function_name is not None
            else None
        ),
    )


def create_patch(orig_func_name, new_func=None, create_dummy=False):
    """
    Create a patch state dictionary for a target function.

    This replaces the original class-based Patch object with a functional
    state container.

    Args:
        orig_func_name (str): Fully qualified function path
                              (e.g., "module.sub.func").
        new_func (callable, optional): Replacement function.
                                       If None, a dummy wrapper is used.
        create_dummy (bool): Whether to create missing modules/functions.

    Returns:
        dict: Patch state dictionary.
    """
    split_name = orig_func_name.rsplit(".", 1)
    if len(split_name) == 1:
        orig_module_name, orig_func_name = orig_func_name, None
    else:
        orig_module_name, orig_func_name = split_name

    if new_func is None:
        new_func = dummy_function_wrapper(orig_func_name)

    state = {
        "orig_module_name": orig_module_name,
        "orig_func_name": orig_func_name,
        "orig_module": None,
        "orig_func": None,
        "patch_func": None,
        "wrappers": [],
        "is_applied": False,
        "create_dummy": create_dummy,
    }

    set_patch_func(state, new_func)

    return state


def set_patch_func(state, new_func, force_patch=False):
    """
    Set or update the patch function in a patch state.

    If the provided function name ends with "wrapper" or "decorator",
    it will be treated as a wrapper and appended to the wrapper chain.

    Args:
        state (dict): Patch state dictionary.
        new_func (callable): New patch or wrapper function.
        force_patch (bool): Whether to overwrite an existing patch function.

    Raises:
        RuntimeError: If a patch already exists and force_patch=False.
    """
    if hasattr(new_func, "__name__") and new_func.__name__.endswith(
        ("wrapper", "decorator")
    ):
        state["wrappers"].append(new_func)
    else:
        if state["patch_func"] and not force_patch:
            raise RuntimeError(f"the patch of {state['orig_func_name']} exist !")
        state["patch_func"] = new_func

    state["is_applied"] = False


def apply_patch(state):
    """
    Apply a single patch to the runtime environment.

    This function:
        1. Resolves the original module and function.
        2. Builds the final patched function (including wrappers).
        3. Replaces the function in the original module.
        4. Scans all loaded modules and replaces references with matching id.

    Args:
        state (dict): Patch state dictionary.
    """
    if state["is_applied"]:
        return

    orig_module, orig_func = parse_path(
        state["orig_module_name"],
        state["orig_func_name"],
        state["create_dummy"],
    )

    state["orig_module"] = orig_module
    state["orig_func"] = orig_func

    final_patch_func = state["patch_func"] or orig_func

    for wrapper in state["wrappers"]:
        final_patch_func = wrapper(final_patch_func)

    if state["orig_func_name"] is not None:
        setattr(orig_module, state["orig_func_name"], final_patch_func)

    # Replace global references with identical function object id
    orig_id = id(orig_func)

    for module in sys.modules.copy().values():
        if (
            state["orig_func_name"] is not None
            and hasattr(module, state["orig_func_name"])
            and id(getattr(module, state["orig_func_name"])) == orig_id
        ):
            setattr(module, state["orig_func_name"], final_patch_func)

    state["is_applied"] = True


PATCHES = {}


def register_patch(
    orig_func_name, new_func=None, force_patch=False, create_dummy=False
):
    """
    Register or update a patch for a target function.

    If the patch does not exist, a new patch state is created.
    If it already exists, its patch function is updated.

    Args:
        orig_func_name (str): Fully qualified target function path.
        new_func (callable, optional): Replacement function.
        force_patch (bool): Whether to overwrite existing patch.
        create_dummy (bool): Whether to create missing modules/functions.
    """
    if orig_func_name not in PATCHES:
        PATCHES[orig_func_name] = create_patch(
            orig_func_name,
            new_func,
            create_dummy,
        )
    else:
        set_patch_func(PATCHES[orig_func_name], new_func, force_patch)


def apply_patches():
    """
    Apply all registered patches.

    Iterates over all stored patch states and applies them.
    Safe to call multiple times (idempotent behavior).
    """
    for patch in PATCHES.values():
        apply_patch(patch)


def dataloader_adaptation():
    """
    Register DataLoader-related patches for DP balance adaptation.

    This function registers a lightweight wrapper around PyTorch's
    _pin_memory_loop to inject cross-DP data reordering, plus
    a replacement for Megatron's RerunDataIterator.
    """
    from loongforge.data.dp_balance.pin_memory_hook import (
        pin_memory_loop_wrapper,
    )
    from loongforge.data.dp_balance.rerun_iterator import (
        RerunDataIterator,
    )

    register_patch(
        "torch.utils.data._utils.pin_memory._pin_memory_loop",
        pin_memory_loop_wrapper,
    )

    register_patch(
        "megatron.core.rerun_state_machine.RerunDataIterator",
        RerunDataIterator,
        create_dummy=True,
    )


def exec_adaptation():
    """
    Main entry point for executing runtime adaptation.

    This function:
        1. Reads global arguments.
        2. Conditionally registers DataLoader-related patches.
        3. Applies all registered patches.

    Intended to be executed once during program initialization.
    """
    from loongforge.utils import get_args

    args = get_args()
    if args.use_vlm_dp_balance:
        dataloader_adaptation()
    apply_patches()


exec_adaptation()
