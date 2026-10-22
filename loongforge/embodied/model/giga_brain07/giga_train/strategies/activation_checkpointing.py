import torch.nn as nn


def module_auto_wrap_policy(module_names: list, sep: str = '__##__'):
    """Build an auto-wrap policy closure from a list of class name rules.

    Args:
        module_names: A list of class name strings. Each item is either
            - "ClassName" to indicate unlimited wrapping for that class, or
            - "ClassName{sep}N" where N is an integer limiting the number of
              occurrences to wrap for that class (first N only).
        sep: The separator string that splits the class name and the numeric limit.

    Returns:
        A callable ``_wrap(module, recurse, nonwrapped_numel) -> bool`` that
        decides whether to wrap the given module instance.
    """
    # Map: class name -> { 'total': max_allowed (or -1 for unlimited), 'count': seen_so_far }
    module_info_dict = dict()
    for module_name in module_names:
        # Split input like "ClassName__##__3" into ["ClassName", "3"]
        parts = module_name.split(sep)
        if len(parts) == 1:
            # No limit provided: wrap all occurrences of this class
            module_info_dict[parts[0]] = dict(total=-1, count=0)
        elif len(parts) == 2:
            # Enforce that the suffix is numeric and set that as the limit
            assert parts[1].isdigit()
            module_info_dict[parts[0]] = dict(total=int(parts[1]), count=0)
        else:
            assert False

    def _wrap(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        """Policy function consumed by activation-checkpointing auto-wrap
        logic.

        Rules:
        - If ``recurse`` is True: always return True to traverse into children.
        - If ``recurse`` is False: make a wrapping decision for this leaf module
          based on its class name and remaining quota.
        """
        # Always traverse deeper when requested; actual wrap decisions at leaves
        if recurse:
            return True
        module_name = module.__class__.__name__
        if module_name in module_info_dict:
            module_info = module_info_dict[module_name]
            if module_info['total'] < 0:
                return True
            module_info['count'] += 1
            if module_info['count'] <= module_info['total']:
                return True
            return False
        return False

    return _wrap
