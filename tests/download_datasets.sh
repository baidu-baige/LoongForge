#!/bin/bash
set -e

# Install bcecmd
PFS_DIR="/mnt/cfs/BaigeOmniCI"

# Define bcecmd wrapper function with retry mechanism
REAL_BCECMD="/usr/local/bin/bcecmd"
bcecmd() {
    local retries=20
    local count=0
    local wait_time=3
    
    while [ $count -lt $retries ]; do
        # Temporarily disable set -e to handle errors manually
        set +e
        $REAL_BCECMD "$@"
        local status=$?
        set -e
        
        if [ $status -eq 0 ]; then
            return 0
        fi
        
        count=$((count + 1))
        echo "========================================================================"
        echo "[WARNING] bcecmd execution failed (Exit code: $status)."
        echo "Retrying in $wait_time seconds... (Attempt $count/$retries)"
        echo "========================================================================"
        sleep $wait_time
    done
    
    echo "[ERROR] bcecmd failed after $retries attempts."
    exit 1
}

# Parse arguments
DOWNLOAD_MODE="default"

for arg in "$@"; do
    case $arg in
        --default)
        DOWNLOAD_MODE="default"
        ;;
        --optional)
        DOWNLOAD_MODE="optional"
        ;;
        *)
            ;;
    esac
done

checkpoint_dir=${PFS_DIR}/megatron_checkpoint
huggingface_dir=${PFS_DIR}/huggingface.co
datasets_dir=${PFS_DIR}/omni_datasets/BaigeOmni/

mkdir -p $checkpoint_dir $huggingface_dir $datasets_dir

if [ "$DOWNLOAD_MODE" == "default" ]; then
    echo "Running default dataset download..."

    # deepseek_v2_lite
    bcecmd bos sync bos:/ai-data/deepseek-ai/DeepSeek-V2-Lite ${huggingface_dir}/deepseek-ai/DeepSeek-V2-Lite
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/deepseek_v2_lite/pile_test/ ${datasets_dir}/deepseek_v2_lite/pile_test/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/deepseek_v2_lite/tp2pp1ep8  ${checkpoint_dir}/deepseek_v2_lite/ 

    # llama2_7b
    bcecmd bos sync bos:/ai-data/Llama-2-7b-hf-test ${huggingface_dir}/meta-llama/Llama-2-7b-hf
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/llama2/pile_test/ ${datasets_dir}/llama2/pile_test/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llama2_7b/tp1pp1  ${checkpoint_dir}/llama2_7b/

    # llama3_8b
    bcecmd bos sync bos:/ai-data/meta-llama/Meta-Llama-3-8B ${huggingface_dir}/meta-llama/Meta-Llama-3-8B
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llama3_8b/tp2pp2/  ${checkpoint_dir}/llama3_8b/

    # qwen3_14b
    bcecmd bos sync bos:/ai-data/Qwen/Qwen3-14B ${huggingface_dir}/Qwen/Qwen3-14B
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen3/pile_test/ ${datasets_dir}/qwen3/pile_test/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_14b/tp2pp4/  ${checkpoint_dir}/qwen3_14b/

    # qwen2.5_vl_7b
    bcecmd bos sync bos:/ai-data/Qwen/Qwen2.5-VL-7B-Instruct ${huggingface_dir}/Qwen/Qwen2.5-VL-7B-Instruct/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2.5_vl_7b/pretrain/llava_details-minigpt4_3500_formate/wds ${datasets_dir}/qwen2.5_vl_7b/pretrain/llava_details-minigpt4_3500_formate/wds
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2.5_vl_7b/sft/test_packing/test_packed_wds_4k/ ${datasets_dir}/qwen2.5_vl_7b/sft/test_packing/test_packed_wds_4k/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen2.5_vl_7b/tp4pp2vpp2/  ${checkpoint_dir}/qwen2.5_vl_7b/

    # llavaov_1.5_4b
    bcecmd bos sync bos:/ai-data/lmms-lab/LLaVA-OneVision-1.5-4B-stage0 ${huggingface_dir}/LLaVA-OneVision-1.5-4B-stage0/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/llavaov_1.5_4b/test_packed_wds_4k/ ${datasets_dir}/llavaov_1.5_4b/test_packed_wds_4k/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llavaov_1.5_4b/tp2pp2/  ${checkpoint_dir}/llavaov_1.5_4b/

    # internvl2.5_8b
    bcecmd bos sync bos:/ai-data/InternVL2_5-8B ${huggingface_dir}/internvl/InternVL2_5-8B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_image ${datasets_dir}/internvl2.5_8b/webdataset_image
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_8b/tp4pp2vpp2/  ${checkpoint_dir}/internvl2.5_8b/

    # internvl3.5_30b_a3b
    bcecmd bos sync bos:/ai-data/OpenGVLab/InternVL3_5-30B-A3B ${huggingface_dir}/internvl/InternVL3_5-30B-A3B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_video ${datasets_dir}/internvl3.5_30b_a3b/webdataset_video
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_30b_a3b/tp2pp2ep4/  ${checkpoint_dir}/internvl3.5_30b_a3b/

    # offline packing
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/offline_packing/test_wds/ ${datasets_dir}/offline_packing_data/
elif [ "$DOWNLOAD_MODE" == "optional" ]; then
    echo "Running optional dataset download..."
    
    # internvl2.5_8b
    bcecmd bos sync bos:/ai-data/InternVL2_5-8B ${huggingface_dir}/internvl/InternVL2_5-8B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_image ${datasets_dir}/internvl2.5_8b/webdataset_image
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_8b/tp4pp1/  ${checkpoint_dir}/internvl2.5/internvl2.5_8b/

    # internvl2.5_26b  
    bcecmd bos sync bos:/ai-data/OpenGVLab/InternVL2_5-26B ${huggingface_dir}/internvl/InternVL2_5-26B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_26b/tp4pp1/  ${checkpoint_dir}/internvl2.5/internvl2.5_26b/

    # internvl2.5_38b
    bcecmd bos sync bos:/ai-data/InternVL2_5-38B ${huggingface_dir}/internvl/InternVL2_5-38B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_38b/tp4pp2/  ${checkpoint_dir}/internvl2.5/internvl2.5_38b/
    
    # internvl2.5_78b
    bcecmd bos sync bos:/ai-data/OpenGVLab/InternVL2_5-78B ${huggingface_dir}/internvl/InternVL2_5-78B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_78b/tp8pp4/  ${checkpoint_dir}/internvl2.5/internvl2.5_78b/

    # internvl3.5_8b
    bcecmd bos sync bos:/ai-data/jhc/OpenGVLab/InternVL3_5-8B ${huggingface_dir}/internvl/InternVL3_5-8B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_8b/tp4pp1/  ${checkpoint_dir}/internvl3.5/internvl3.5_8b/

    # internvl3.5_14b
    bcecmd bos sync bos:/ai-data/OpenGVLab/InternVL3_5-14B ${huggingface_dir}/internvl/InternVL3_5-14B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_14b/tp4pp1/  ${checkpoint_dir}/internvl3.5/internvl3.5_14b/
    
    # internvl3.5_30b_a3b
    bcecmd bos sync bos:/ai-data/OpenGVLab/InternVL3_5-30B-A3B ${huggingface_dir}/internvl/InternVL3_5-30B-A3B/ 
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_30b_a3b/tp4pp1ep8etp1/  ${checkpoint_dir}/internvl3.5/internvl3.5_30b_a3b/

    # internvl3.5_38b
    bcecmd bos sync bos:/ai-data/OpenGVLab/InternVL3_5-38B ${huggingface_dir}/internvl/InternVL3_5-38B
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_38b/tp4pp2/  ${checkpoint_dir}/internvl3.5/internvl3.5_38b/

    # qwen2_7b
    bcecmd bos sync bos:/ai-data/huggingface.co/Qwen/Qwen2-7B-Instruct ${huggingface_dir}/Qwen/Qwen2-7B-Instruct/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2/pile_test/ ${datasets_dir}/qwen2/pile_test/

    # qwen2.5_7b
    bcecmd bos sync bos:/ai-data/Qwen2.5-7B-Instruct ${huggingface_dir}/Qwen/Qwen2.5-7B-Instruct/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen/pile_test/ ${datasets_dir}/qwen2.5/pile_test/

    # qwen2.5_vl_7b
    bcecmd bos sync bos:/ai-data/Qwen/Qwen2.5-VL-7B-Instruct ${huggingface_dir}/Qwen/Qwen2.5-VL-7B-Instruct/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen2.5_vl_7b/tp1pp1/  ${checkpoint_dir}/qwen2.5_vl/qwen2.5_vl_7b/

    # qwen3_8b
    bcecmd bos sync bos:/ai-data/Qwen/Qwen3-8B ${huggingface_dir}/Qwen/Qwen3-8B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_8b/tp1pp1/  ${checkpoint_dir}/qwen3/qwen3_8b/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen3/pile_test/ ${datasets_dir}/qwen3/pile_test/

    # qwen3_30b_a3b
    bcecmd bos sync bos:/ai-data/Qwen/Qwen3-30B-A3B ${huggingface_dir}/Qwen/Qwen3-30B-A3B/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_30b_a3b/tp2pp2ep4/  ${checkpoint_dir}/qwen3/qwen3_30b_a3b/

    # qwen3_vl_30b_a3b
    bcecmd bos sync bos:/ai-data/Qwen3-VL-30B-A3B-Instruct ${huggingface_dir}/Qwen/Qwen3-VL-30B-A3B-Instruct/
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen_vl/qwen3vl_30b_a3b_data/ ${datasets_dir}/qwen3_vl/qwen3vl_30b_a3b_data
    bcecmd bos sync bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_vl_30b_a3b/tp1pp2ep8/  ${checkpoint_dir}/qwen3_vl/qwen3_vl_30b_a3b/

fi

echo "========================================"
echo "All Datasets Downloaded Successfully!"
echo "========================================"
