""" bias_dropout_add """

from typing import Optional, Tuple
import torch

from megatron.core.jit import jit_fuser


def _bias_dropout_add_func(x_with_bias, gate, residual, prob, training):
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).

    x, bias = x_with_bias  # unpack

    # If we want to train mixed precision, then the output of this function
    # should be half precision. However, in AMP O1, the input (residual) is
    # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
    # GPU communication to hang. Therefore, we need to cast residual to the same
    # dtype as x.
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

    # The Dropout operation, Residual Addition and the tensor returning can be
    # done generically outside the if statement, but that stops fusing of Bias
    # Addition-Dropout-Residual Addition operation. So doing it together inside
    # the conditional branch to improve performance
    if bias is not None:
        x = x + bias
    if gate is not None:
        x = gate * x
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bias_dropout_add_unfused(training):
    """ bias_dropout_add_unfused """
    def _bias_dropout_add(x_with_bias, gate, residual, prob):
        return _bias_dropout_add_func(x_with_bias, gate, residual, prob, training)

    return _bias_dropout_add


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], gate: torch.Tensor, residual: torch.Tensor, prob: float,
) -> torch.Tensor:
    """ bias_dropout_add_fused_train """
    return _bias_dropout_add_func(x_with_bias, gate, residual, prob, True)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], gate: torch.Tensor, residual: torch.Tensor, prob: float,
) -> torch.Tensor:
    """ bias_dropout_add_fused_inference """
    return _bias_dropout_add_func(x_with_bias, gate, residual, prob, False)


def get_bias_dropout_add(training, fused):
    """ get_bias_dropout_add """
    if fused:
        # jit scripting for a nn.module (with dropout) is not
        # triggering the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if training:
            return bias_dropout_add_fused_train
        else:
            return bias_dropout_add_fused_inference
    else:
        return bias_dropout_add_unfused(training)
