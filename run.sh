# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/freeze_lr_1e5/
  
CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
  --save_folder ./weights/lr_1e3_resize_image_rgb_relu/ \
  --resume_epoch 160 \
  --resume_net ./weights/lr_1e3_resize_image_rgb_relu/mobilenet0.25_epoch_160.pth
  
