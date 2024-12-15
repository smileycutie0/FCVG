CUDA_VISIBLE_DEVICES=0 python demo_FCVG.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --controlnext_path checkpoints/controlnext.safetensors \
  --unet_path checkpoints/unet.safetensors \
  --image1_path example/real/001/00.png \
  --image2_path example/real/001/24.png \
  --output_dir results \
  --control_weight 1.0 \
  --num_inference_steps 25 \
  --height 576 \
  --width 1024 \


