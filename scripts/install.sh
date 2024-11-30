#!/usr/bin/env bash
echo "Deleting ComfyUI"
rm -rf /workspace/ComfyUI

echo "Deleting venv"
rm -rf /workspace/venv

echo "Cloning ComfyUI repo to /workspace"
cd /workspace
git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git

echo "Installing Ubuntu updates"
apt update
apt -y upgrade

echo "Creating and activating venv"
cd ComfyUI
python -m venv /workspace/venv
source /workspace/venv/bin/activate

echo "Installing Torch"
pip install --no-cache-dir torch==2.1.2+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing xformers"
pip3 install --no-cache-dir xformers==0.0.23.post1+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "Installing ComfyUI"
pip3 install -r requirements.txt

echo "Installing ComfyUI Manager"

# Array of repositories to clone
repos=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/XLabs-AI/x-flux-comfyui"
    "https://github.com/giriss/comfy-image-saver"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/city96/ComfyUI-GGUF"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/fofr/ComfyUI-Impact-Pack"
    "https://github.com/ltdrdata/ComfyUI-Inspire-Pack"
    "https://github.com/theUpsider/ComfyUI-Logic"
    "https://github.com/Acly/comfyui-tooling-nodes"
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/XLabs-AI/x-flux-comfyui"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/melMass/comfy_mtb"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/kijai/ComfyUI-DepthAnythingV2"
)

# Install custom nodes
for repo in "${repos[@]}"; do
    repo_name=$(basename "$repo")
    git clone "$repo" "custom_nodes/$repo_name"
    
    if [ -f "custom_nodes/$repo_name/requirements.txt" ]; then
        cd "custom_nodes/$repo_name"
        pip3 install -r requirements.txt
        cd ../..
    fi
done

echo "Installing RunPod Serverless dependencies"
pip3 install huggingface_hub runpod

# Function to download with HF token
download_hf() {
    local url="$1"
    local output="${2:-$(basename "$url")}"
    wget --header="Authorization: Bearer ${HUGGINGFACE_TOKEN}" "$url" -O "$output"
}

# Function to create directory and cd into it
create_and_cd() {
    mkdir -p "$1"
    cd "$1"
}

# Download models
echo "Downloading Flux Models"
create_and_cd "/workspace/ComfyUI/models/unet"
download_hf "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
download_hf "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf"
download_hf "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors"

echo "Downloading Flux Style Models"
create_and_cd "/workspace/ComfyUI/models/style_models"
download_hf "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors"

echo "Downloading VAE"
create_and_cd "/workspace/ComfyUI/models/vae"
download_hf "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors"
download_hf "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"

echo "Downloading Clip Models"
create_and_cd "/workspace/ComfyUI/models/clip"
download_hf "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
download_hf "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
download_hf "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors" "clip-vit-large-patch14.safetensors"

echo "Downloading Clip Vision"
create_and_cd "/workspace/ComfyUI/models/clip_vision"
download_hf "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors"

echo "Downloading Lora Models"
create_and_cd "/workspace/ComfyUI/models/lora"
declare -A lora_models=(
    ["Flux_Aquarell_Watercolor_v2.safetensors"]="https://huggingface.co/SebastianBodza/Flux_Aquarell_Watercolor_v2/resolve/main/lora.safetensors"
    ["XLabs-AI-flux-RealismLora.safetensors"]="https://huggingface.co/XLabs-AI/flux-RealismLora/resolve/main/lora.safetensors"
    ["flux_realism_lora.safetensors"]="https://huggingface.co/comfyanonymous/flux_RealismLora_converted_comfyui/resolve/main/flux_realism_lora.safetensors"
    ["anime_lora_comfy_converted.safetensors"]="https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/anime_lora_comfy_converted.safetensors"
    ["art_lora_comfy_converted.safetensors"]="https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/art_lora_comfy_converted.safetensors"
    ["disney_lora_comfy_converted.safetensors"]="https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/disney_lora_comfy_converted.safetensors"
    ["mjv6_lora_comfy_converted.safetensors"]="https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/mjv6_lora_comfy_converted.safetensors"
    ["realism_lora_comfy_converted.safetensors"]="https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/realism_lora_comfy_converted.safetensors"
    ["scenery_lora_comfy_converted.safetensors"]="https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/scenery_lora_comfy_converted.safetensors"
    ["strangerzonehf-Flux-Super-Realism-LoRA.safetensors"]="https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2/resolve/main/lora.safetensors"
    ["flux_dev_frostinglane_araminta_k.safetensors"]="https://huggingface.co/alvdansen/frosting_lane_flux/resolve/main/flux_dev_frostinglane_araminta_k.safetensors"
    ["FLUX-dev-lora-Logo-Design.safetensors"]="https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design/resolve/main/FLUX-dev-lora-Logo-Design.safetensors"
    ["super-realism.safetensors"]="https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA/resolve/main/super-realism.safetensors"
    ["strangerzonehf-Flux-Animeo-v1-LoRA.safetensors"]="https://huggingface.co/strangerzonehf/Flux-Animeo-v1-LoRA/resolve/main/Animeo.safetensors"
    ["brushpenbob-FLUX_MidJourney_Anime.safetensors"]="https://huggingface.co/brushpenbob/flux-midjourney-anime/resolve/main/FLUX_MidJourney_Anime.safetensors"
    ["Canopus-Anime-Character-Art-FluxDev-LoRA.safetensors"]="https://huggingface.co/prithivMLmods/Canopus-LoRA-Flux-Anime/resolve/main/Canopus-Anime-Character-Art-FluxDev-LoRA.safetensors"
    ["aleksa-codes-flux-ghibsky-illustration.safetensors"]="https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/resolve/main/lora.safetensors"
)

for output_file in "${!lora_models[@]}"; do
    download_hf "${lora_models[$output_file]}" "$output_file"
done

echo "Downloading Upscalers"
create_and_cd "/workspace/ComfyUI/models/upscale_models"
download_hf "https://huggingface.co/ashleykleynhans/upscalers/resolve/main/4x-UltraSharp.pth"
download_hf "https://huggingface.co/ashleykleynhans/upscalers/resolve/main/lollypop.pth"

echo "Creating log directory"
mkdir -p /workspace/logs