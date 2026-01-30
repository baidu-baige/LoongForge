# Heterogeneous Tensor Parallel

AIAK-Training-Omni supports heterogeneous Tensor Parallel (TP) configuration for encoder and decoder, meaning encoder and decoder can use different TP sizes for parallel computation.

In this design, encoder and decoder are treated as two sub-modules with different computational characteristics and resource requirements. The system allows them to configure independent tensor parallel groups separately, enabling more fine-grained parallel strategy control within the same training or inference task.

This heterogeneous TP mechanism enables the model to flexibly select the most appropriate parallel granularity based on differences in computational intensity, parameter scale, activation size, and communication patterns between encoder and decoder, rather than being constrained by a unified TP configuration for the entire model.

![heterogeneous_tp](../../../assets/_images/tp_pic1.png)

## 1. Usage Method
Set `tensor-model-parallel-size` in the corresponding model's vit.yaml to specify the vit tp size. For example, adding `tensor_model_parallel_size: 2` in qwen3_vit specifies the vit's tp size:

```yaml
_target_: aiak_training_omni.models.encoder.Qwen3VisionModelConfig

num_layers: 27
hidden_size: 1152
kv_channels: 72
ffn_hidden_size: 4304
patch_size: 16
num_attention_heads: 16
num_query_groups: 16
image_size: [1344, 1344]
activation_func: ${act:gelu}
normalization: "LayerNorm"
add_bias_linear: true
add_qkv_bias: true
swiglu: False
group_query_attention: False
gated_linear_unit: False
position_embedding_type: "none"
bias_activation_fusion: False
deepstack_visual_indexes: [8, 16, 24]
num_position_embeddings: 2304

tensor_model_parallel_size: 2

model_type: "qwen3_vit"
```

Specify decoder tp size in the corresponding shell script:

```bash
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    # --sequence-parallel
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
)
```

## 2. Performance Results
Based on qwen2.5vl7b testing with decoder tp = 4 and encoder tp of 1, 2, and 4, different settings show different performance characteristics. Specific performance for different models requires testing.

![heterogeneous_tp_res](../../../assets/_images/tp_pic2.png)

For small-scale encoders like Vit (0.6b), a 5% performance improvement was achieved in the qwen2.5vl 7b model.