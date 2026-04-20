# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Transform LoongForge checkpoint keys back to vanilla naming schemes."""

def transform_key_forward(key, forward_mappings):
    """
    Transform key using forward mapping
    
    Args:
        key: Key to transform
        forward_mappings: Forward mapping dictionary
        
    Returns:
        Transformed key
    """
    return _transform_key_with_mappings(key, forward_mappings)


def transform_key_reverse(key, reverse_mappings):
    """
    Transform key using reverse mapping
    
    Args:
        key: Key to transform
        reverse_mappings: Reverse mapping dictionary
        
    Returns:
        Transformed key
    """
    return _transform_key_with_mappings(key, reverse_mappings)


def _transform_key_with_mappings(key, mappings):
    """
    Internal helper function for key transformation
    
    Args:
        key: Key to transform
        mappings: Mapping dictionary
        
    Returns:
        Transformed key if mapping found, original key otherwise
    """
    for old_prefix, new_prefix in mappings.items():
        if key.startswith(old_prefix + '.'):
            rest = key[len(old_prefix) + 1:]
            return f"{new_prefix}.{rest}"
    return key