def format_size(x: int, factor: int = 1000) -> str:
    """Format number of parameters into human-readable string.

    Args:
        x: Number of parameters.
        factor: Factor to convert to larger units (1000 for K, 1000000 for M, 1000000000 for G).

    Returns:
        str: Formatted string like "1.2G", "1200K", or "1200000".
    """
    if x > 1e8:
        return '{:.1f}G'.format(x / (factor * factor * factor))
    if x > 1e5:
        return '{:.1f}M'.format(x / (factor * factor))
    if x > 1e2:
        return '{:.1f}K'.format(x / factor)
    return str(x)


def parameter_count(model, factor: int = 1000):
    """Count total number of parameters in a model and format as human-readable string.

    Args:
        model: PyTorch model to count parameters of.
        factor: Factor to convert to larger units (1000 for K, 1000000 for M, 1000000000 for G).

    Returns:
        tuple: (total_params, formatted_size)
    """
    count = 0
    for parameter in model.parameters():
        count += parameter.numel()
    return count, format_size(count, factor)
