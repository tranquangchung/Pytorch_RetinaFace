# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/freeze_lr_1e5/
  
CUDA_VISIBLE_DEVICES=0 python train_quantization.py --network mobile0.25 \
  --save_folder ./weights/lr_1e3_resize_image_rgb_relu_conv_quantization/ \
  --resume_net ./weights/lr_1e3_resize_image_rgb_relu_conv/mobilenet0.25_epoch_170.pth 
  
# CUDA_VISIBLE_DEVICES=0 python load_quantization.py --network mobile0.25 \
#   --save_folder ./weights/lr_1e3_resize_image_rgb_relu/ \
#   --resume_net ./weights/mobilenet0.25_epoch_240.pth 
  
