from typing import Any, Callable, Iterable

import torch

from .build import OPTIMIZERS


@OPTIMIZERS.register
class CAME8Bit(torch.optim.Optimizer):
    """8-bit implementation of the CAME optimizer.

    Args:
        params (iterable): Parameters to optimize
        lr (float): Learning rate
        eps (tuple[float, float]): Numerical stability constants
        clip_threshold (float): Gradient clipping threshold
        betas (tuple[float, float, float]): Momentum coefficients
        weight_decay (float): Weight decay
        block_size (int): Quantization block size, larger blocks are more memory efficient but less precise
        min_8bit_size (int): Minimum parameter size to use 8-bit, only layers larger than this will be quantized

    Note:
        1. Only large Linear and 1x1 Conv layers are quantized to 8-bit
        2. All statistics (e.g., exp_avg_sq_row) remain in 32-bit for stability
        3. Uses a simple min-max quantization strategy, each block is quantized separately
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float | None = None,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        block_size: int = 2048,  # Quantization block size
        min_8bit_size: int = 16384,  # Minimum parameter size to use 8-bit
    ) -> None:
        """Initialize the 8-bit CAME optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate (> 0).
            eps: Numerical stability terms used in update and residual.
            clip_threshold: Global RMS-based gradient clipping threshold.
            betas: Momentum coefficients for first, second, residual moments.
            weight_decay: L2 weight decay factor.
            block_size: Block size for quantizing optimizer states.
            min_8bit_size: Only parameters larger than this are quantized.
        """
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            block_size=block_size,
            min_8bit_size=min_8bit_size,
        )
        super().__init__(params, defaults)

    def _should_use_8bit(self, param_shape: torch.Size) -> bool:
        """Determines whether parameters should be quantized to 8-bit.

        Rules:
        1. Linear layers: parameter count > min_8bit_size
        2. 1x1 conv layers: parameter count > min_8bit_size
        3. Other cases: use 32-bit
        """
        if len(param_shape) == 2:  # Linear layers
            return param_shape[0] * param_shape[1] > self.defaults['min_8bit_size']
        elif len(param_shape) == 4 and param_shape[2] == 1 and param_shape[3] == 1:  # Only quantize 1x1 conv
            return param_shape[0] * param_shape[1] > self.defaults['min_8bit_size']
        return False  # Other layers are not quantized

    def _quantize_state(self, state_tensor: torch.Tensor, block_size: int = 2048):
        """Quantizes the state tensor to 8-bit.

        Args:
            state_tensor: Tensor to be quantized
            block_size: Block size for quantization

        Returns:
            List of quantized data blocks, each block contains:
            - data: uint8 data
            - scale: Quantization scale
            - min: Minimum value
        """
        if state_tensor.numel() <= 1:
            return state_tensor

        quantized_chunks = []
        for chunk in state_tensor.split(block_size):
            # Calculate quantization parameters
            chunk_min = chunk.min()
            chunk_max = chunk.max()
            scale = (chunk_max - chunk_min) / 255

            # Quantize to 0-255 range
            quantized_chunk = ((chunk - chunk_min) / scale).round().byte()
            quantized_chunks.append({'data': quantized_chunk, 'scale': scale, 'min': chunk_min})
        return quantized_chunks

    def _dequantize_state(self, quantized_chunks):
        """Dequantizes 8-bit quantized data to 32-bit floats.

        Args:
            quantized_chunks: List of quantized data blocks

        Returns:
            Dequantized 32-bit float tensor
        """
        if not isinstance(quantized_chunks, list):
            return quantized_chunks

        chunks = []
        for chunk_dict in quantized_chunks:
            # Dequantize: value = data * scale + min
            chunk = chunk_dict['data'].float() * chunk_dict['scale'] + chunk_dict['min']
            chunks.append(chunk)
        return torch.cat(chunks)

    def _dequantize_state_first_step(self, quantized_chunks):
        """Efficient dequantization specifically for the first step."""
        if not isinstance(quantized_chunks, list):
            return quantized_chunks

        # 1. Dequantize all chunks to CPU first
        dequantized_chunks = []
        for chunk_dict in quantized_chunks:
            chunk = chunk_dict['data'].float() * chunk_dict['scale'] + chunk_dict['min']
            dequantized_chunks.append(chunk)
            # Clear original data
            del chunk_dict['data']
            torch.cuda.empty_cache()

        # 2. Concatenate all chunks
        result = torch.cat(dequantized_chunks)

        # 3. Clear intermediate results
        del dequantized_chunks
        torch.cuda.empty_cache()

        return result

    def _get_options(self, param_shape: torch.Size) -> tuple[bool, str]:
        if len(param_shape) == 4:  # Convolutional layer
            if param_shape[2] == 1 and param_shape[3] == 1:  # 1x1 conv
                return True, '1x1_conv'
            else:  # 3x3 conv or others
                return False, 'conv'
        elif len(param_shape) == 2:  # Linear layer
            return True, 'linear'
        return False, 'other'

    def _rms(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        """Performs a single optimization step.

        Main steps:
        1. Determine whether 8-bit quantization is needed
        2. Update first and second moment estimates
        3. Calculate update step size
        4. Apply confidence-guided strategy
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError('CAME8bit does not support sparse gradients.')

                state = self.state[p]
                grad_shape = grad.shape
                factored, layer_type = self._get_options(grad_shape)

                # Determine whether to use 8-bit quantization
                use_8bit = self._should_use_8bit(grad_shape)

                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Only use 8-bit quantization for large matrices
                    if use_8bit:
                        state['exp_avg'] = self._quantize_state(torch.zeros_like(grad), group['block_size'])
                    else:
                        state['exp_avg'] = torch.zeros_like(grad)

                    if factored:
                        if layer_type == '1x1_conv' or layer_type == 'linear':
                            # Row and column statistics remain in 32-bit
                            if isinstance(grad, torch.distributed.tensor.DTensor):
                                kwargs = dict(dtype=grad.dtype, device_mesh=grad.device_mesh, placements=grad.placements)
                                state['exp_avg_sq_row'] = torch.distributed.tensor.zeros(grad_shape[0], **kwargs)
                                state['exp_avg_sq_col'] = torch.distributed.tensor.zeros(grad_shape[1], **kwargs)
                                state['exp_avg_res_row'] = torch.distributed.tensor.zeros(grad_shape[0], **kwargs)
                                state['exp_avg_res_col'] = torch.distributed.tensor.zeros(grad_shape[1], **kwargs)
                            else:
                                state['exp_avg_sq_row'] = torch.zeros(grad_shape[0]).type_as(grad)
                                state['exp_avg_sq_col'] = torch.zeros(grad_shape[1]).type_as(grad)
                                state['exp_avg_res_row'] = torch.zeros(grad_shape[0]).type_as(grad)
                                state['exp_avg_res_col'] = torch.zeros(grad_shape[1]).type_as(grad)
                        else:
                            if use_8bit:
                                state['exp_avg_sq'] = self._quantize_state(torch.zeros_like(grad), group['block_size'])
                            else:
                                state['exp_avg_sq'] = torch.zeros_like(grad)
                    else:
                        if use_8bit:
                            state['exp_avg_sq'] = self._quantize_state(torch.zeros_like(grad), group['block_size'])
                        else:
                            state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['RMS'] = 0

                state['step'] += 1
                state['RMS'] = self._rms(p.data)

                exp_avg = self._dequantize_state(state['exp_avg']) if use_8bit else state['exp_avg']

                update = (grad**2) + group['eps'][0]
                if factored:
                    # Row and column decomposition case
                    exp_avg_sq_row = state['exp_avg_sq_row']  # 32-bit
                    exp_avg_sq_col = state['exp_avg_sq_col']  # 32-bit

                    if layer_type == '1x1_conv' or layer_type == 'linear':
                        if len(grad_shape) == 4:
                            update_reshaped = update.squeeze(-1).squeeze(-1)
                        else:
                            update_reshaped = update

                        # Update row and column statistics
                        exp_avg_sq_row.mul_(group['betas'][1]).add_(update_reshaped.mean(dim=1), alpha=1.0 - group['betas'][1])
                        exp_avg_sq_col.mul_(group['betas'][1]).add_(update_reshaped.mean(dim=0), alpha=1.0 - group['betas'][1])

                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    if layer_type == '1x1_conv':
                        update = update.view(grad_shape[0], grad_shape[1], 1, 1)
                    update.mul_(grad)
                else:
                    # Non-decomposition case
                    exp_avg_sq = self._dequantize_state(state['exp_avg_sq']) if use_8bit else state['exp_avg_sq']
                    exp_avg_sq.mul_(group['betas'][1]).add_(update, alpha=1.0 - group['betas'][1])
                    if use_8bit:
                        state['exp_avg_sq'] = self._quantize_state(exp_avg_sq, group['block_size'])
                    else:
                        state['exp_avg_sq'] = exp_avg_sq
                    update = exp_avg_sq.rsqrt().mul_(grad)

                # Gradient clipping
                update.div_((self._rms(update) / group['clip_threshold']).clamp_(min=1.0))

                # Update first moment
                exp_avg.mul_(group['betas'][0]).add_(update, alpha=1 - group['betas'][0])

                # Re-quantize (if needed)
                if use_8bit:
                    state['exp_avg'] = self._quantize_state(exp_avg, group['block_size'])
                else:
                    state['exp_avg'] = exp_avg

                # Confidence-guided strategy
                res = (update - exp_avg) ** 2 + group['eps'][1]

                if factored:
                    exp_avg_res_row = state['exp_avg_res_row']  # 32-bit
                    exp_avg_res_col = state['exp_avg_res_col']  # 32-bit

                    if layer_type == '1x1_conv' or layer_type == 'linear':
                        if len(grad_shape) == 4:
                            res_reshaped = res.squeeze(-1).squeeze(-1)
                        else:
                            res_reshaped = res

                        # Update residual statistics
                        exp_avg_res_row.mul_(group['betas'][2]).add_(res_reshaped.mean(dim=1), alpha=1.0 - group['betas'][2])
                        exp_avg_res_col.mul_(group['betas'][2]).add_(res_reshaped.mean(dim=0), alpha=1.0 - group['betas'][2])

                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    if layer_type == '1x1_conv':
                        res_approx = res_approx.view(grad_shape[0], grad_shape[1], 1, 1)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                # Apply update
                update.mul_(group['lr'])
                p.data.add_(-update)

        return loss

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads the state dictionary and converts the corresponding states to
        8-bit."""
        super().load_state_dict(state_dict)  # Call the parent class method

        for state in self.state.values():
            for key in [
                'exp_avg',
                'exp_avg_sq',
                'exp_avg_sq_row',
                'exp_avg_sq_col',
                'exp_avg_res_row',
                'exp_avg_res_col',
            ]:
                if key in state:
                    if isinstance(state[key], list):
                        state[key] = [
                            {
                                'data': exp['data'].byte(),  # Directly convert data to 8-bit
                                'scale': exp['scale'],  # Keep scale unchanged
                                'min': exp['min'],  # Keep min unchanged
                            }
                            for exp in state[key]
                        ]
                    elif isinstance(state[key], torch.Tensor):
                        # If it's a tensor, keep it as 32-bit
                        state[key] = state[key].float()  # Ensure it's 32-bit

        del state_dict
        torch.cuda.empty_cache()
