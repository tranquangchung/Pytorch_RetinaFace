# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/freeze_lr_1e5/
  
CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
  --save_folder ./weights/lr_1e3_resize_360_640 \
  # --resume_net ./weights/mobilenet0.25_Final.pth 
