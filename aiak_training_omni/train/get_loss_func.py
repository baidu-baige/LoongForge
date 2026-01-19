"""Generic loss function"""
from aiak_training_omni.utils import get_args
import torch
import os
from megatron.core import mpu
import torch.distributed as dist
from megatron.training.utils import average_losses_across_data_parallel_group


def default_loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    loss_weight: torch.Tensor = None,
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """
    args = get_args()

    if (loss_weight is not None and loss_weight.sum() == 0) or (loss_mask.sum() == 0):
        output_tensor = output_tensor * 0.0
        valid_mask = False
    else:
        valid_mask = True
    
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    if loss_weight is not None:
        shift_weights = loss_weight.view(-1)
        shift_weights_sum = shift_weights.sum()
        if (
            args.loss_reduction_all_gather and args.context_parallel_size > 1
        ):  # TODO: check args.loss_reduction_all_gather
            torch.distributed.all_reduce(
                shift_weights_sum,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            shift_weights_sum = shift_weights_sum / mpu.get_data_parallel_world_size(
                with_context_parallel=True
            )
        loss = torch.sum(losses * shift_weights)
    else:
        loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    if valid_mask:
        if loss_weight is not None:
            num_tokens = shift_weights_sum.clone().detach().to(torch.int)
        else:
            num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    else:
        num_tokens = 1

    # Reduce loss for logging.
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    loss_reduced_dict = {'lm loss': reporting_loss if not args.legacy_reporting_loss_reduction
                         else reporting_loss[0] / num_tokens}

    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(
            num_input_tokens, dtype=torch.int, device=output_tensor.device
        )
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = (
            input_tokens * args.context_parallel_size
        )

    return (loss, num_tokens, loss_reduced_dict)


def loss_func_internvl(
    loss_mask: torch.Tensor, loss_weight: torch.Tensor, output_tensor: torch.Tensor
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """
    args = get_args()

    valid_mask = True
    if (loss_weight is not None and loss_weight.sum() == 0) or (loss_mask.sum() == 0):
        valid_mask = False
        output_tensor = output_tensor * 0.0  # skip update current microbatch

    losses = output_tensor.view(-1).float()  # [B, s]
    loss_mask = loss_mask.view(-1).float()  # [B * s]

    if loss_weight is not None:
        shift_weights = loss_weight.view(-1)
        shift_weights_sum = shift_weights.sum()
        if args.loss_reduction_all_gather:
            torch.distributed.all_reduce(
                shift_weights_sum,
                op=dist.ReduceOp.SUM,
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            shift_weights_sum = shift_weights_sum / mpu.get_data_parallel_world_size(
                with_context_parallel=True
            )
        loss = torch.sum(losses * shift_weights)
    else:
        loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # reduce loss for logging.
    if valid_mask:
        if loss_weight is not None:
            num_tokens = shift_weights_sum.clone().detach().to(torch.int)
        else:
            num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    else:
        num_tokens = 1

    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    loss_reduced_dict = {'lm loss': reporting_loss if not args.legacy_reporting_loss_reduction
                         else reporting_loss[0] / num_tokens}

    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(
            num_input_tokens, dtype=torch.int, device=output_tensor.device
        )
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = (
            input_tokens * args.context_parallel_size
        )

    return loss, num_tokens, loss_reduced_dict
