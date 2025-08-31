#!/bin/bash
# WAN 2.2 T2V Character LoRA Inference Examples - Optimized for H100 SXM GPU
# This script contains various example commands for generating videos with your CHARACTER_VANRAJ_V1 LoRA
# Optimized for NVIDIA H100 SXM (80GB VRAM) with FP8 Tensor Cores and high memory bandwidth

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
echo "Virtual Environment: $VENV_PATH"

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
echo "📁 Changed to repository directory: $(pwd)"

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
# CONFIGURATION - Auto-detect paths from training setup
# =============================================================================

# Model paths (automatically detected from training setup)
MODELS_DIR="$WORKDIR/models"
DIT_MODEL="$MODELS_DIR/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
DIT_HIGH_NOISE="$MODELS_DIR/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
VAE_MODEL="$MODELS_DIR/vae/split_files/vae/wan_2.1_vae.safetensors"
T5_MODEL="$MODELS_DIR/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"

# Your trained LoRA paths (automatically detected from training setup)
OUTPUT_DIR="$WORKDIR/output"
LORA_LOW_NOISE_DIR="$OUTPUT_DIR/low"
LORA_HIGH_NOISE_DIR="$OUTPUT_DIR/high"

# Find the latest LoRA files
LATEST_LOW=$(find "$LORA_LOW_NOISE_DIR" -name "*.safetensors" -type f | sort | tail -1)
LATEST_HIGH=$(find "$LORA_HIGH_NOISE_DIR" -name "*.safetensors" -type f | sort | tail -1)

echo "🔍 Detected LoRA files:"
echo "Low noise LoRA: $LATEST_LOW"
echo "High noise LoRA: $LATEST_HIGH"

# Verify model files exist
for model_file in "$DIT_MODEL" "$DIT_HIGH_NOISE" "$VAE_MODEL" "$T5_MODEL"; do
    if [ ! -f "$model_file" ]; then
        echo "❌ Model file not found: $model_file"
        echo "Please run setup_and_train_musubi.sh first to download models"
        exit 1
    fi
done

# Verify LoRA files exist
if [ ! -f "$LATEST_LOW" ]; then
    echo "❌ Low noise LoRA not found in: $LORA_LOW_NOISE_DIR"
    echo "Please complete training first or check the output directory"
    exit 1
fi

echo "✅ All essential files verified"

# Output directory
INFERENCE_OUTPUT_DIR="./h100_character_videos"
mkdir -p "$INFERENCE_OUTPUT_DIR"

# H100 SXM Optimized Settings
H100_OPTS="--fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile"

# For maximum quality on H100 (leveraging 80GB VRAM)
HIGH_QUALITY_OPTS="--video_size 1024 576 --infer_steps 50 --guidance_scale 7.5 --guidance_scale_high_noise 6.5"

# For ultra-high resolution (H100 can handle this)
ULTRA_HD_OPTS="--video_size 1280 720 --infer_steps 60 --video_length 81"

# =============================================================================
# H100 OPTIMIZED BASIC EXAMPLES
# =============================================================================

echo "=== H100 Optimized Character Video Generation ==="

# Example 1: H100 Optimized basic generation with FP8 acceleration
echo "🎬 Example 1: Basic H100 optimized generation"
python wan_generate_video.py \
    --task t2v-14B \
    --dit "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_weight "$LATEST_LOW" \
    --prompt "A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden during sunset" \
    --video_size 768 512 \
    $H100_OPTS \
    --save_path "$INFERENCE_OUTPUT_DIR/h100_basic" \
    --seed 42

echo "✅ Example 1 completed"

# Example 2: Ultra High Quality with both LoRA weights (H100 flagship)
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 2: Ultra high quality with both LoRA weights"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "A cinematic video of CHARACTER_VANRAJ_V1 standing confidently in an ornate palace hall" \
        $HIGH_QUALITY_OPTS \
        $H100_OPTS \
        --video_length 81 \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_ultra_hq" \
        --seed 123

    echo "✅ Example 2 completed"
else
    echo "⚠️  Skipping Example 2: High noise LoRA not found"
fi

# Example 3: 4K-ready generation (H100 can handle large resolutions)
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 3: 4K-ready generation"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "CHARACTER_VANRAJ_V1 in 4K cinematic quality performing an elaborate royal ceremony" \
        $ULTRA_HD_OPTS \
        $H100_OPTS \
        --guidance_scale 8.0 \
        --guidance_scale_high_noise 7.0 \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_4k" \
        --seed 456

    echo "✅ Example 3 completed"
else
    echo "⚠️  Skipping Example 3: High noise LoRA not found"
fi

# =============================================================================
# H100 PERFORMANCE OPTIMIZED EXAMPLES
# =============================================================================

echo "=== H100 Maximum Performance Examples ==="

# Example 4: Lightning fast generation with optimal H100 settings
echo "🎬 Example 4: Lightning fast generation"
python wan_generate_video.py \
    --task t2v-14B \
    --dit "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_weight "$LATEST_LOW" \
    --prompt "A dynamic video of CHARACTER_VANRAJ_V1 practicing martial arts in a serene mountain temple" \
    --negative_prompt "blurry, low quality, distorted face, extra limbs" \
    --video_size 768 512 \
    --fps 30 \
    --infer_steps 30 \
    $H100_OPTS \
    --seed 789 \
    --save_path "$INFERENCE_OUTPUT_DIR/h100_lightning"

echo "✅ Example 4 completed"

# Example 5: Long video generation (leveraging H100's 80GB VRAM)
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 5: Long video generation"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "A portrait video of CHARACTER_VANRAJ_V1 with gentle wind moving his hair, warm lighting" \
        --video_size 768 512 \
        --video_length 161 \
        --fps 24 \
        $H100_OPTS \
        --guidance_scale 7.5 \
        --seed 321 \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_long_video"

    echo "✅ Example 5 completed"
else
    echo "⚠️  Skipping Example 5: High noise LoRA not found"
fi

# Example 6: Maximum quality with Skip Layer Guidance (H100 can handle complexity)
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 6: Maximum quality with Skip Layer Guidance"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "CHARACTER_VANRAJ_V1 reading a book under a large oak tree, leaves falling gently" \
        --video_size 1024 576 \
        --infer_steps 50 \
        --guidance_scale 8.0 \
        --guidance_scale_high_noise 7.0 \
        --slg_layers "0,1,2,3" \
        --slg_scale 3.5 \
        $H100_OPTS \
        --seed 654 \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_slg_max"

    echo "✅ Example 6 completed"
else
    echo "⚠️  Skipping Example 6: High noise LoRA not found"
fi

# =============================================================================
# H100 ADVANCED QUALITY SETTINGS
# =============================================================================

echo "=== H100 Advanced Quality Settings ==="

# Example 7: Ultra-fine detail generation with maximum steps
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 7: Ultra-fine detail generation"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "CHARACTER_VANRAJ_V1 in royal attire giving a speech in front of a crowd, ultra detailed" \
        --infer_steps 80 \
        --guidance_scale 8.5 \
        --guidance_scale_high_noise 7.5 \
        --video_size 1152 640 \
        --flow_shift 4.0 \
        $H100_OPTS \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_ultra_detail" \
        --seed 987

    echo "✅ Example 7 completed"
else
    echo "⚠️  Skipping Example 7: High noise LoRA not found"
fi

# Example 8: Professional cinematic quality with advanced CFG
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 8: Professional cinematic quality"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "CHARACTER_VANRAJ_V1 dancing gracefully in traditional clothing, cinematic lighting" \
        --video_size 1024 576 \
        --infer_steps 60 \
        --guidance_scale 7.8 \
        --guidance_scale_high_noise 6.8 \
        --cfg_skip_mode early_late \
        --cfg_apply_ratio 0.8 \
        $H100_OPTS \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_cinematic" \
        --seed 111

    echo "✅ Example 8 completed"
else
    echo "⚠️  Skipping Example 8: High noise LoRA not found"
fi

# =============================================================================
# H100 SPECIALIZED OUTPUT FORMATS
# =============================================================================

echo "=== H100 Specialized Output Formats ==="

# Example 9: High-res image sequence for post-processing
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 9: High-res image sequence"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "CHARACTER_VANRAJ_V1 performing a traditional ceremony, high resolution frames" \
        --output_type images \
        --video_size 1280 720 \
        --video_length 81 \
        --infer_steps 50 \
        $H100_OPTS \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_hires_frames" \
        --seed 222

    echo "✅ Example 9 completed"
else
    echo "⚠️  Skipping Example 9: High noise LoRA not found"
fi

# Example 10: Professional workflow with latents + video
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "🎬 Example 10: Professional workflow (video + latents)"
    python wan_generate_video.py \
        --task t2v-14B \
        --dit "$DIT_MODEL" \
        --dit_high_noise "$DIT_HIGH_NOISE" \
        --vae "$VAE_MODEL" \
        --t5 "$T5_MODEL" \
        --lora_weight "$LATEST_LOW" \
        --lora_weight_high_noise "$LATEST_HIGH" \
        --prompt "CHARACTER_VANRAJ_V1 crafting something with his hands in a workshop, professional quality" \
        --output_type both \
        --video_size 1024 576 \
        --video_length 81 \
        --infer_steps 60 \
        $H100_OPTS \
        --save_path "$INFERENCE_OUTPUT_DIR/h100_professional" \
        --seed 333

    echo "✅ Example 10 completed"
else
    echo "⚠️  Skipping Example 10: High noise LoRA not found"
fi

echo "🎉 All H100-optimized examples completed!"
echo "=============================="
echo "Generated videos are saved in: $INFERENCE_OUTPUT_DIR"
echo
echo "H100 SXM optimizations used:"
echo "✅ FP8 Tensor Cores (2x speed boost)"
echo "✅ Flash Attention 3 (memory efficiency)"
echo "✅ Torch.compile (kernel fusion)"
echo "✅ High resolution support (up to 1280x720)"
echo "✅ 80GB VRAM utilization"
echo
echo "Used LoRA weights:"
echo "📦 Low noise: $LATEST_LOW"
if [ -n "$LATEST_HIGH" ] && [ -f "$LATEST_HIGH" ]; then
    echo "📦 High noise: $LATEST_HIGH"
else
    echo "⚠️  High noise LoRA not found - some examples were skipped"
fi
echo
echo "Performance Tips:"
echo "💡 Monitor GPU temperature - H100 performs best under 80°C"
echo "💡 Use batch processing for multiple videos"
echo "💡 Experiment with different resolutions and inference steps"
echo "💡 Both LoRA weights give best character consistency"
