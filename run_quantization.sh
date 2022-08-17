# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/freeze_lr_1e5/
  
CUDA_VISIBLE_DEVICES=1 python train_quantization.py --network mobile0.25 \
  --save_folder ./weights/lr_1e3_resize_360_640_RGB_NORM_01_transfer_learning_quantization_L2 \
  --resume_net ./weights/lr_1e5_resize_360_640_RGB_NORM_01_transfer_learning/mobilenet0.25_Final.pth \
  # --lr 1e-5
  
# CUDA_VISIBLE_DEVICES=0 python load_quantization.py --network mobile0.25 \
#   --save_folder ./weights/lr_1e3_resize_image_rgb_relu/ \
#   --resume_net ./weights/mobilenet0.25_epoch_240.pth 
  
