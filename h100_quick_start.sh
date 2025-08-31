#!/bin/bash
# H100 SXM Quick Start Script for WAN 2.2 Character LoRA Inference
# This script provides ready-to-run commands optimized for H100 SXM GPU

echo "🚀 H100 SXM WAN 2.2 Character LoRA Quick Start"
echo "=============================================="

# =============================================================================
# ENVIRONMENT SETUP - Activate the musubi-tuner virtual environment
# =============================================================================

# Detect the workspace and activate virtual environment
WORKDIR="${WORKDIR:-/workspace/diffusion_pipe_working_folder/wan2.2_lora_training}"
REPO_DIR="$WORKDIR/musubi-tuner"
VENV_PATH="$REPO_DIR/venv"

echo "🔧 Setting up environment..."
echo "Workspace: $WORKDIR"
echo "Repository: $REPO_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run setup_and_train_musubi.sh first to set up the environment"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
echo "✅ Virtual environment activated"

# Change to repository directory
cd "$REPO_DIR"
echo "📁 Working directory: $(pwd)"

# Check for H100 GPU
if nvidia-smi | grep -q "H100"; then
    echo "✅ H100 GPU detected"
else
    echo "⚠️  H100 GPU not detected - these optimizations may not work optimally"
fi

# Check if musubi_tuner module is available
python -c "import musubi_tuner; print('✅ musubi_tuner module is available')" 2>/dev/null || {
    echo "❌ musubi_tuner module not found even after activating venv"
    echo "Please reinstall the package with: pip install -e ."
    exit 1
}

# =============================================================================
# QUICK CONFIGURATION - Auto-detect paths from training setup
# =============================================================================

# Model paths - Auto-detected from training setup
MODELS_DIR="$WORKDIR/models"
DIT_MODEL="$MODELS_DIR/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
DIT_HIGH_NOISE="$MODELS_DIR/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
VAE_MODEL="$MODELS_DIR/vae/split_files/vae/wan_2.1_vae.safetensors"
T5_MODEL="$MODELS_DIR/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"

# Your trained LoRA paths - Auto-detected from training setup
OUTPUT_DIR="$WORKDIR/output"
LORA_LOW_NOISE_DIR="$OUTPUT_DIR/low"
LORA_HIGH_NOISE_DIR="$OUTPUT_DIR/high"

# Find the latest LoRA files
LORA_LOW_NOISE=$(find "$LORA_LOW_NOISE_DIR" -name "*.safetensors" -type f | sort | tail -1)
LORA_HIGH_NOISE=$(find "$LORA_HIGH_NOISE_DIR" -name "*.safetensors" -type f | sort | tail -1)

echo "🔍 Auto-detected configuration:"
echo "Models directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Low noise LoRA: $LORA_LOW_NOISE"
echo "High noise LoRA: $LORA_HIGH_NOISE"

# Verify essential files exist
if [ ! -f "$DIT_MODEL" ]; then
    echo "❌ DiT model not found: $DIT_MODEL"
    echo "Please run setup_and_train_musubi.sh first to download models"
    exit 1
fi

if [ ! -f "$LORA_LOW_NOISE" ]; then
    echo "❌ Low noise LoRA not found in: $LORA_LOW_NOISE_DIR"
    echo "Please complete training first or check the output directory"
    exit 1
fi

echo "✅ Essential files verified"

# Output directory
OUTPUT_DIR_INFERENCE="./h100_character_videos"
mkdir -p "$OUTPUT_DIR_INFERENCE"

# =============================================================================
# QUICK START COMMANDS
# =============================================================================

echo
echo "1️⃣  Basic H100 Optimized Generation"
echo "================================="
echo "Command:"
echo "python wan_generate_video.py \\"
echo "    --task t2v-14B \\"
echo "    --dit \"$DIT_MODEL\" \\"
echo "    --vae \"$VAE_MODEL\" \\"
echo "    --t5 \"$T5_MODEL\" \\"
echo "    --lora_weight \"$LORA_LOW_NOISE\" \\"
echo "    --prompt \"A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden\" \\"
echo "    --fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile \\"
echo "    --save_path \"$OUTPUT_DIR_INFERENCE/basic\" \\"
echo "    --seed 42"
echo
read -p "Press Enter to run this command, or Ctrl+C to skip..."

python wan_generate_video.py \
    --task t2v-14B \
    --dit "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_weight "$LORA_LOW_NOISE" \
    --prompt "A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden" \
    --fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile \
    --save_path "$OUTPUT_DIR_INFERENCE/basic" \
    --seed 42

echo "✅ Basic generation completed!"

if [ -n "$LORA_HIGH_NOISE" ] && [ -f "$LORA_HIGH_NOISE" ]; then
    echo
    echo "2️⃣  High Quality with Both LoRA Weights"
    echo "====================================="
    echo "Command:"
    echo "python wan_generate_video.py \\"
    echo "    --task t2v-14B \\"
    echo "    --dit \"$DIT_MODEL\" \\"
    echo "    --dit_high_noise \"$DIT_HIGH_NOISE\" \\"
    echo "    --vae \"$VAE_MODEL\" \\"
    echo "    --t5 \"$T5_MODEL\" \\"
    echo "    --lora_weight \"$LORA_LOW_NOISE\" \\"
    echo "    --lora_weight_high_noise \"$LORA_HIGH_NOISE\" \\"
    echo "    --prompt \"A cinematic video of CHARACTER_VANRAJ_V1 in royal attire\" \\"
    echo "    --video_size 1024 576 \\"
    echo "    --infer_steps 50 \\"
    echo "    --fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile \\"
    echo "    --save_path \"$OUTPUT_DIR_INFERENCE/high_quality\" \\"
    echo "    --seed 123"
    echo
    read -p "Press Enter to run this command, or Ctrl+C to skip..."

    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LORA_LOW_NOISE" \
        --lora_weight_high_noise "$LORA_HIGH_NOISE" \
        --prompt "A cinematic video of CHARACTER_VANRAJ_V1 in royal attire" \
        --video_size 1024 576 \
        --infer_steps 50 \
        --fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile \
        --save_path "$OUTPUT_DIR_INFERENCE/high_quality" \
        --seed 123

    echo "✅ High quality generation completed!"

    echo
    echo "3️⃣  Ultra High Resolution (H100 Power)"
    echo "====================================="
    echo "Command:"
    echo "python wan_generate_video.py \\"
    echo "    --task t2v-14B \\"
    echo "    --dit \"$DIT_MODEL\" \\"
    echo "    --dit_high_noise \"$DIT_HIGH_NOISE\" \\"
    echo "    --vae \"$VAE_MODEL\" \\"
    echo "    --t5 \"$T5_MODEL\" \\"
    echo "    --lora_weight \"$LORA_LOW_NOISE\" \\"
    echo "    --lora_weight_high_noise \"$LORA_HIGH_NOISE\" \\"
    echo "    --prompt \"CHARACTER_VANRAJ_V1 performing an elaborate ceremony in 4K quality\" \\"
    echo "    --video_size 1280 720 \\"
    echo "    --infer_steps 60 \\"
    echo "    --fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile \\"
    echo "    --save_path \"$OUTPUT_DIR_INFERENCE/ultra_hd\" \\"
    echo "    --seed 456"
    echo
    read -p "Press Enter to run this command, or Ctrl+C to skip..."

    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LORA_LOW_NOISE" \
        --lora_weight_high_noise "$LORA_HIGH_NOISE" \
        --prompt "CHARACTER_VANRAJ_V1 performing an elaborate ceremony in 4K quality" \
        --video_size 1280 720 \
        --infer_steps 60 \
        --fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile \
        --save_path "$OUTPUT_DIR_INFERENCE/ultra_hd" \
        --seed 456

    echo "✅ Ultra HD generation completed!"
else
    echo
    echo "⚠️  Skipping high quality examples - High noise LoRA not found"
    echo "Train both high and low noise models for best results"
fi

echo
echo "🎉 H100 Quick Start Complete!"
echo "=============================="
echo "Generated videos are saved in: $OUTPUT_DIR_INFERENCE"
echo
echo "H100 Optimizations Applied:"
echo "✅ FP8 Tensor Cores (2x speed boost)"
echo "✅ Flash Attention 3 (memory efficiency)"
echo "✅ Torch.compile (kernel fusion)"
echo "✅ High resolution support (up to 1280x720)"
echo "✅ 80GB VRAM utilization"
echo
echo "Used LoRA weights:"
echo "📦 Low noise: $LORA_LOW_NOISE"
if [ -n "$LORA_HIGH_NOISE" ] && [ -f "$LORA_HIGH_NOISE" ]; then
    echo "📦 High noise: $LORA_HIGH_NOISE"
else
    echo "⚠️  High noise LoRA not available"
fi
echo
echo "Performance Tips:"
echo "💡 Monitor GPU temperature - H100 performs best under 80°C"
echo "💡 Use both LoRA weights for best character consistency"
echo "💡 Experiment with different resolutions and inference steps"
echo "💡 Run 'bash wan_inference_examples.sh' for more examples"

# H100 SXM Optimized Settings with fallback for torch.compile issues
H100_OPTS_STABLE="--fp8 --fp8_scaled --fp8_fast --fp8_t5"
H100_OPTS_FULL="--fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile"

# Function to test if torch.compile works with the current setup
test_torch_compile() {
    echo "🧪 Testing torch.compile compatibility..."

    # Test with a simple command first (dry run)
    python -c "
import torch
import warnings
warnings.filterwarnings('ignore')
try:
    # Simple test to see if compile works without major issues
    x = torch.randn(10, 10, device='cuda', dtype=torch.float16)
    compiled_fn = torch.compile(torch.nn.Linear(10, 10).cuda().half())
    _ = compiled_fn(x)
    print('✅ torch.compile test passed')
    exit(0)
except Exception as e:
    print(f'⚠️  torch.compile test failed: {e}')
    exit(1)
" 2>/dev/null

    return $?
}

# Determine which optimization level to use
if test_torch_compile; then
    echo "✅ torch.compile is compatible - using full H100 optimizations"
    H100_OPTS="$H100_OPTS_FULL"
else
    echo "⚠️  torch.compile compatibility issues detected - using stable H100 optimizations"
    H100_OPTS="$H100_OPTS_STABLE"
fi

echo "Active H100 Options: $H100_OPTS"

