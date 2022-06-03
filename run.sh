# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/freeze_lr_1e5/
  
CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
  --save_folder ./weights/lr_1e3_resize_image_norm_rgb/ \
  --resume_epoch 140
  
