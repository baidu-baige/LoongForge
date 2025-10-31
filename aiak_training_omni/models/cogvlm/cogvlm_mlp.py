""" mlp for cogvlm."""
from megatron.core.transformer.mlp import MLP

class CogvlmMlp(MLP):
    """
    mlp for cogvlm.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, hidden_states, **kwargs):
        """ forward """
        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states, **kwargs)

        # activation function
        intermediate_parallel = self.activation_func(intermediate_parallel, bias_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel, **kwargs)

        return output, output_bias
