#!/bin/bash
# WAN 2.2 T2V Character LoRA Inference Examples - Optimized for H100 SXM GPU
# This script contains various example commands for generating videos with your CHARACTER_VANRAJ_V1 LoRA
# Optimized for NVIDIA H100 SXM (80GB VRAM) with FP8 Tensor Cores and high memory bandwidth

# =============================================================================
# CONFIGURATION - Update these paths according to your setup
# =============================================================================

# Model paths (update these to your actual model paths)
DIT_MODEL="/path/to/wan2.2/dit_model.safetensors"
DIT_HIGH_NOISE="/path/to/wan2.2/dit_high_noise_model.safetensors"  # Optional
VAE_MODEL="/path/to/wan2.2/vae_model.safetensors"
T5_MODEL="/path/to/wan2.2/t5_model.safetensors"

# Your trained LoRA paths
LORA_LOW_NOISE="/path/to/your/low_noise_lora.safetensors"
LORA_HIGH_NOISE="/path/to/your/high_noise_lora.safetensors"

# Output directory
OUTPUT_DIR="./character_videos"

# H100 SXM Optimized Settings
# Enable all H100-specific optimizations
H100_OPTS="--fp8 --fp8_scaled --fp8_fast --fp8_t5 --compile --attn_mode flash3"

# For maximum quality on H100 (leveraging 80GB VRAM)
HIGH_QUALITY_OPTS="--video_size 1024 576 --infer_steps 50 --guidance_scale 7.5 --guidance_scale_high_noise 6.5"

# For ultra-high resolution (H100 can handle this)
ULTRA_HD_OPTS="--video_size 1280 720 --infer_steps 60 --video_length 81"

# =============================================================================
# H100 OPTIMIZED BASIC EXAMPLES
# =============================================================================

echo "=== H100 Optimized Character Video Generation ==="

# Example 1: H100 Optimized basic generation with FP8 acceleration
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden during sunset" \
    --video_size 768 512 \
    $H100_OPTS \
    --save_path "$OUTPUT_DIR/h100_basic" \
    --seed 42 \
    --execute

# Example 2: Ultra High Quality with both LoRA weights (H100 flagship)
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "A cinematic video of CHARACTER_VANRAJ_V1 standing confidently in an ornate palace hall" \
    $HIGH_QUALITY_OPTS \
    $H100_OPTS \
    --video_length 81 \
    --save_path "$OUTPUT_DIR/h100_ultra_hq" \
    --seed 123 \
    --execute

# Example 3: 4K-ready generation (H100 can handle large resolutions)
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 in 4K cinematic quality performing an elaborate royal ceremony" \
    $ULTRA_HD_OPTS \
    $H100_OPTS \
    --guidance_scale 8.0 \
    --guidance_scale_high_noise 7.0 \
    --save_path "$OUTPUT_DIR/h100_4k" \
    --seed 456 \
    --execute

# =============================================================================
# H100 PERFORMANCE OPTIMIZED EXAMPLES
# =============================================================================

echo "=== H100 Maximum Performance Examples ==="

# Example 4: Lightning fast generation with optimal H100 settings
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "A dynamic video of CHARACTER_VANRAJ_V1 practicing martial arts in a serene mountain temple" \
    --negative_prompt "blurry, low quality, distorted face, extra limbs" \
    --video_size 768 512 \
    --fps 30 \
    --infer_steps 30 \
    $H100_OPTS \
    --seed 789 \
    --save_path "$OUTPUT_DIR/h100_lightning" \
    --execute

# Example 5: Long video generation (leveraging H100's 80GB VRAM)
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "A portrait video of CHARACTER_VANRAJ_V1 with gentle wind moving his hair, warm lighting" \
    --video_size 768 512 \
    --video_length 161 \
    --fps 24 \
    $H100_OPTS \
    --guidance_scale 7.5 \
    --seed 321 \
    --save_path "$OUTPUT_DIR/h100_long_video" \
    --execute

# Example 6: Maximum quality with Skip Layer Guidance (H100 can handle complexity)
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 reading a book under a large oak tree, leaves falling gently" \
    --video_size 1024 576 \
    --infer_steps 50 \
    --guidance_scale 8.0 \
    --guidance_scale_high_noise 7.0 \
    --slg_layers "0,1,2,3" \
    --slg_scale 3.5 \
    $H100_OPTS \
    --seed 654 \
    --save_path "$OUTPUT_DIR/h100_slg_max" \
    --execute

# =============================================================================
# H100 ADVANCED QUALITY SETTINGS
# =============================================================================

echo "=== H100 Advanced Quality Settings ==="

# Example 7: Ultra-fine detail generation with maximum steps
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 in royal attire giving a speech in front of a crowd, ultra detailed" \
    --infer_steps 80 \
    --guidance_scale 8.5 \
    --guidance_scale_high_noise 7.5 \
    --video_size 1152 640 \
    --flow_shift 4.0 \
    $H100_OPTS \
    --save_path "$OUTPUT_DIR/h100_ultra_detail" \
    --seed 987 \
    --execute

# Example 8: Professional cinematic quality with advanced CFG
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 dancing gracefully in traditional clothing, cinematic lighting" \
    --video_size 1024 576 \
    --infer_steps 60 \
    --guidance_scale 7.8 \
    --guidance_scale_high_noise 6.8 \
    --cfg_skip_mode early_late \
    --cfg_apply_ratio 0.8 \
    $H100_OPTS \
    --save_path "$OUTPUT_DIR/h100_cinematic" \
    --seed 111 \
    --execute

# =============================================================================
# H100 BATCH PROCESSING EXAMPLES
# =============================================================================

echo "=== H100 Batch Processing (Leveraging 80GB VRAM) ==="

# Example 9: Batch generation with optimized memory usage
python wan_generate_video.py \
    --dit "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_weight "$LORA_LOW_NOISE" \
    --lora_weight_high_noise "$LORA_HIGH_NOISE" \
    --from_file character_prompts.txt \
    --video_size 768 512 \
    --infer_steps 40 \
    --guidance_scale 7.5 \
    --guidance_scale_high_noise 6.5 \
    $H100_OPTS \
    --save_path "$OUTPUT_DIR/h100_batch" \
    --task t2v-14B

# =============================================================================
# H100 SPECIALIZED OUTPUT FORMATS
# =============================================================================

echo "=== H100 Specialized Output Formats ==="

# Example 10: High-res image sequence for post-processing
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 performing a traditional ceremony, high resolution frames" \
    --output_type images \
    --video_size 1280 720 \
    --video_length 81 \
    --infer_steps 50 \
    $H100_OPTS \
    --save_path "$OUTPUT_DIR/h100_hires_frames" \
    --seed 222 \
    --execute

# Example 11: Professional workflow with latents + video
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 crafting something with his hands in a workshop, professional quality" \
    --output_type both \
    --video_size 1024 576 \
    --video_length 81 \
    --infer_steps 60 \
    $H100_OPTS \
    --save_path "$OUTPUT_DIR/h100_professional" \
    --seed 333 \
    --execute

# =============================================================================
# H100 EXPERIMENTAL HIGH-END FEATURES
# =============================================================================

echo "=== H100 Experimental High-End Features ==="

# Example 12: Ultra-long high-quality video (pushing H100 limits)
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "An epic journey of CHARACTER_VANRAJ_V1 through changing seasons and landscapes" \
    --video_size 1024 576 \
    --video_length 241 \
    --fps 24 \
    --infer_steps 50 \
    --guidance_scale 7.5 \
    $H100_OPTS \
    --save_path "$OUTPUT_DIR/h100_epic_journey" \
    --seed 444 \
    --execute

echo "All H100-optimized examples completed! Check the output directories for your generated videos."
echo "H100 SXM optimizations used:"
echo "- FP8 Tensor Cores for 2x speed improvement"
echo "- Flash Attention 3 for memory efficiency"
echo "- Torch.compile for kernel fusion"
echo "- Large resolution support (up to 1280x720)"
echo "- Extended video lengths (up to 241 frames)"
echo "- Advanced quality settings leveraging 80GB VRAM"
