# Quick Start: VLM Model Training
This document will guide you through the process of quickly starting the training of a Vision-Language Model (VLM) under the AIAK-Training-Omni framework.

# 1. Environment Preparation
## 1.1 Code Download
Make sure you have downloaded the code for this repository and the dependent AIAK-Megatron codebase. Assume they are located in the same directory level:



```
# Assuming the current directory is /workspace
git clone <git_repo_url>/AIAK-Training-Omni.git
git clone <git_repo_url>/AIAK-Megatron.git  
```
## 1.2 Install Dependencies
Enter the AIAK-Training-Omni directory to install the necessary Python dependencies:



```
cd AIAK-Training-Omni
pip install -r requirements.txt
```


Also, set the environment variables:



```
export AIAK_TRAINING_PATH=/workspace/AIAK-Training-Omni
export MEGATRON_PATH=/workspace/AIAK-Megatron
export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH
```


# 2. Prepare Model Weights
## 2.1 Download Hugging Face Model
### 2.2 Convert Weight Format
Training usually starts with open-source Hugging Face weights. We need to download the weights first and then convert them to the format supported by this framework (Megatron-Core format).

## 2.1 Download Hugging Face Model
Take InternVL2.5-8B as an example, please download the model weights from Hugging Face.

### 2.2 Convert Weight Format
Use the conversion script in the examples directory to convert the HF format to the Mcore format.

For example, to convert qwen2.5_vl_7B:



```
# Enter the conversion script directory
cd $AIAK_TRAINING_PATH/examples/qwen2.5_vl/checkpoint_convert

# Edit the script convert_qwen2.5_vl_7b_mcore_to_hf.sh to configure the LOAD and SAVE paths
# LOAD: Points to the downloaded HF model path
# SAVE: Points to the output path after conversion
vim convert_qwen2.5_vl_7b_mcore_to_hf.sh

# Execute the conversion
bash convert_qwen2.5_vl_7b_mcore_to_hf.sh
```


# 3. Prepare Data
## 3.1 Raw Data
It is recommended to prepare the data in JSON format, with each item containing the image path and corresponding multi-turn conversation content.

**Example Data (data.json):**



```
{
  "id": 64,
  "image": [
    "mmdu-45k_pics/Bk1Gh1Y1S2EChGZzde_1_1.jpg",
    "mmdu-45k_pics/Bk1GhVk1RhkB7W80RCTo_1.jpg"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "Image1: <image>\nImage2: <image>\n### Questions:\nQ: Describe the architectural style observed in Image1 and compare it to the architectural style seen in Image2.\nQ: How do the entrances of the buildings in Image1 and Image2 differ in terms of design and decoration?\nQ: What historical or cultural significance might be attached to the buildings seen in Image1 and Image2?\nQ: Analyze the role of religious structures such as those depicted in Image1 and Image2 within their respective communities.\nQ: What emotions or atmospheres are portrayed by the surroundings and structures in Image1 compared to Image2?\nQ: Discuss the possible symbolic meanings of any elements or features present in Image1 and Image2.\nQ: What activities or actions typically associated with buildings like those in Image1 and Image2 can you infer from their appearance?\nQ: How might the relationship between the religious buildings and the community differ between what is seen in Image1 and Image2?\nQ: Identify and compare any vegetation or natural elements surrounding the buildings in Image1 and Image2.\nQ: How do the settings of the buildings in Image1 and Image2 compare in terms of environment and landscape?"
    },
    {
      "from": "gpt",
      "value": "The architectural styles observed in Image1 and Image2 display distinct characteristics that reflect their unique identities and cultural contexts. Image1 showcases a Gothic Revival architectural style, as evident from its pointed arches, intricate window tracery, and the imposing vertical emphasis typical of Gothic church architecture. The building's façade is symmetrical, with a central nave flanked by two towers. The use of brick and stone highlights the structure’s robust and sturdy construction, evoking a sense of permanence and endurance. The tall windows are adorned with detailed carvings, showcasing the artisanship and craftsmanship that went into the creation of this building. This style is often associated with medieval cathedrals in Europe, aiming to inspire awe and reverence through its grandiosity and intricate detailing.\n\nIn contrast, Image2 depicts a more modern and less ornate architectural style. The structure appears to draw inspiration from both contemporary and traditional elements. Its sloping roofline and the sleek, pointed tower indicate a departure from the heavily ornate features of traditional styles, leaning towards simplicity and functionality. The use of lighter materials, possibly concrete or stucco, gives the building a bright and airy appearance. Large, clear windows allow natural light to flood the interior, symbolizing transparency and openness.\n\nEach style is a product of its time and place, reflecting the prevailing attitudes, materials available, and architectural trends. Gothic Revival in Image1 emphasizes historical continuity and spiritual grandeur, while the modern/contemporary style in Image2 suggests a forward-looking approach, embracing simplicity and light. Despite their differences, both buildings serve the primary function of a religious sanctuary, providing spaces for worship and community gathering."
    }
  ]
}
```


## 3.2 Convert to WebDataset Format
* **Tool Path**: tools/data_preprocess/vlm/
* **Usage Guide**: Please refer to the VLM data preprocessing documentation

After the conversion is complete, the path in the output_dir is the path that needs to be filled in for the DATA_PATH in the training script.

# 4. Start Training (SFT)
Take the SFT (Supervised Fine-Tuning) training of Qwen2_5_vl_7b as an example.

## 4.1 Modify the Training Script
Open examples/qwen2.5_vl/sft/sft_qwen2_5_vl_7b.sh and modify the following path configurations as needed:

```
# Dataset path
DATA_PATH=/path/to/your/dataset

# Tokenizer path (usually the HF model directory)
TOKENIZER_PATH=/path/to/hf/model

# Pre-trained weight loading path (the Mcore weight path converted in the previous step)
CHECKPOINT_LOAD_PATH=/path/to/converted/mcore/checkpoint

# Training result save path
CHECKPOINT_SAVE_PATH=/path/to/save/checkpoints
```


## 4.2 Run Training
```
# In the examples/internvl2.5/ directory
bash sft_internvl2_5_8b.sh
```


## 4.3 Monitor Logs
The script will output TensorBoard logs to the directory specified by TENSORBOARD_PATH by default. You can view the training curves through TensorBoard.

# 5. Common Issues
* **Module Not Found**: Please check if the PYTHONPATH correctly includes AIAK-Training-Omni and AIAK-Megatron.
* **OOM (Out of Memory)**: Try reducing the global_batch_size or enabling checkpoint-activation (recompute).