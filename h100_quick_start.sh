#!/bin/bash
# H100 SXM Quick Start Script for WAN 2.2 Character LoRA Inference
# This script provides ready-to-run commands optimized for H100 SXM GPU

echo "🚀 H100 SXM WAN 2.2 Character LoRA Quick Start"
echo "=============================================="

# Check for H100 GPU
if nvidia-smi | grep -q "H100"; then
    echo "✅ H100 GPU detected"
else
    echo "⚠️  H100 GPU not detected - these optimizations may not work optimally"
fi

# =============================================================================
# QUICK CONFIGURATION - EDIT THESE PATHS
# =============================================================================

# Model paths - UPDATE THESE TO YOUR ACTUAL PATHS
DIT_MODEL="/workspace/models/wan2.2/dit_model.safetensors"
DIT_HIGH_NOISE="/workspace/models/wan2.2/dit_high_noise_model.safetensors"
VAE_MODEL="/workspace/models/wan2.2/vae_model.safetensors"
T5_MODEL="/workspace/models/wan2.2/t5_model.safetensors"

# Your trained LoRA paths - UPDATE THESE
LORA_LOW_NOISE="/workspace/loras/CHARACTER_VANRAJ_V1_low_noise.safetensors"
LORA_HIGH_NOISE="/workspace/loras/CHARACTER_VANRAJ_V1_high_noise.safetensors"

# Output directory
OUTPUT_DIR="./h100_character_videos"

# =============================================================================
# QUICK START COMMANDS
# =============================================================================

echo
echo "1️⃣  Basic H100 Optimized Generation"
echo "================================="
echo "Command:"
echo "python wan_inference_script.py \\"
echo "    --dit_model \"$DIT_MODEL\" \\"
echo "    --vae \"$VAE_MODEL\" \\"
echo "    --t5 \"$T5_MODEL\" \\"
echo "    --lora_low_noise \"$LORA_LOW_NOISE\" \\"
echo "    --prompt \"A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden\" \\"
echo "    --save_path \"$OUTPUT_DIR/basic\" \\"
echo "    --execute"
echo
read -p "Press Enter to run this command, or Ctrl+C to skip..."

python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden" \
    --save_path "$OUTPUT_DIR/basic" \
    --execute

echo
echo "2️⃣  High Quality with Both LoRA Weights"
echo "====================================="
echo "Command:"
echo "python wan_inference_script.py \\"
echo "    --dit_model \"$DIT_MODEL\" \\"
echo "    --dit_high_noise \"$DIT_HIGH_NOISE\" \\"
echo "    --vae \"$VAE_MODEL\" \\"
echo "    --t5 \"$T5_MODEL\" \\"
echo "    --lora_low_noise \"$LORA_LOW_NOISE\" \\"
echo "    --lora_high_noise \"$LORA_HIGH_NOISE\" \\"
echo "    --prompt \"A cinematic video of CHARACTER_VANRAJ_V1 in royal attire\" \\"
echo "    --video_size 1024 576 \\"
echo "    --infer_steps 50 \\"
echo "    --save_path \"$OUTPUT_DIR/high_quality\" \\"
echo "    --execute"
echo
read -p "Press Enter to run this command, or Ctrl+C to skip..."

python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "A cinematic video of CHARACTER_VANRAJ_V1 in royal attire" \
    --video_size 1024 576 \
    --infer_steps 50 \
    --save_path "$OUTPUT_DIR/high_quality" \
    --execute

echo
echo "3️⃣  Ultra High Resolution (H100 Power)"
echo "====================================="
echo "Command:"
echo "python wan_inference_script.py \\"
echo "    --dit_model \"$DIT_MODEL\" \\"
echo "    --dit_high_noise \"$DIT_HIGH_NOISE\" \\"
echo "    --vae \"$VAE_MODEL\" \\"
echo "    --t5 \"$T5_MODEL\" \\"
echo "    --lora_low_noise \"$LORA_LOW_NOISE\" \\"
echo "    --lora_high_noise \"$LORA_HIGH_NOISE\" \\"
echo "    --prompt \"CHARACTER_VANRAJ_V1 performing an elaborate ceremony in 4K quality\" \\"
echo "    --video_size 1280 720 \\"
echo "    --infer_steps 60 \\"
echo "    --save_path \"$OUTPUT_DIR/ultra_hd\" \\"
echo "    --execute"
echo
read -p "Press Enter to run this command, or Ctrl+C to skip..."

python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 performing an elaborate ceremony in 4K quality" \
    --video_size 1280 720 \
    --infer_steps 60 \
    --save_path "$OUTPUT_DIR/ultra_hd" \
    --execute

echo
echo "4️⃣  Batch Processing Multiple Prompts"
echo "==================================="
echo "Command:"
echo "python wan_generate_video.py \\"
echo "    --dit \"$DIT_MODEL\" \\"
echo "    --dit_high_noise \"$DIT_HIGH_NOISE\" \\"
echo "    --vae \"$VAE_MODEL\" \\"
echo "    --t5 \"$T5_MODEL\" \\"
echo "    --lora_weight \"$LORA_LOW_NOISE\" \\"
echo "    --lora_weight_high_noise \"$LORA_HIGH_NOISE\" \\"
echo "    --from_file character_prompts.txt \\"
echo "    --fp8 --fp8_scaled --fp8_fast --fp8_t5 \\"
echo "    --attn_mode flash3 --compile \\"
echo "    --video_size 768 512 \\"
echo "    --save_path \"$OUTPUT_DIR/batch\" \\"
echo "    --task t2v-14B"
echo
read -p "Press Enter to run batch processing, or Ctrl+C to skip..."

python wan_generate_video.py \
    --dit "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_weight "$LORA_LOW_NOISE" \
    --lora_weight_high_noise "$LORA_HIGH_NOISE" \
    --from_file character_prompts.txt \
    --fp8 --fp8_scaled --fp8_fast --fp8_t5 \
    --attn_mode flash3 --compile \
    --video_size 768 512 \
    --save_path "$OUTPUT_DIR/batch" \
    --task t2v-14B

echo
echo "🎉 H100 Quick Start Complete!"
echo "=============================="
echo "Generated videos are saved in: $OUTPUT_DIR"
echo
echo "H100 Optimizations Applied:"
echo "✅ FP8 Tensor Cores (2x speed boost)"
echo "✅ Flash Attention 3 (memory efficiency)"
echo "✅ Torch.compile (kernel fusion)"
echo "✅ High resolution support (up to 1280x720)"
echo "✅ 80GB VRAM utilization"
echo
echo "Performance Tips:"
echo "💡 Monitor GPU temperature - H100 performs best under 80°C"
echo "💡 Use batch processing for multiple videos"
echo "💡 Experiment with different resolutions and inference steps"
echo "💡 Both LoRA weights give best character consistency"
echo
echo "Next Steps:"
echo "📝 Edit character_prompts.txt for custom batch generation"
echo "📝 Try different video sizes: 768x512, 1024x576, 1280x720"
echo "📝 Adjust inference steps: 30 (fast), 50 (balanced), 80 (quality)"
