# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/freeze_lr_1e5/
  
# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/lr_1e3_resize_480_480/ \
#   --resume_net ./weights/lr_1e3_resize_640_640_origin/mobilenet0.25_Final.pth \
#   --lr 1e-5 

# CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
#   --save_folder ./weights/lr_1e3_resize_480_480_L2/ \

CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 \
  --save_folder ./weights/lr_1e5_resize_360_640_RGB_NORM_01_transfer_learning_scale_camera/ \
  --resume_net ./weights/lr_1e3_resize_640_640_origin/mobilenet0.25_Final.pth \
  --lr 1e-5 

# CUDA_VISIBLE_DEVICES=1 python train.py --network mobinetv3 \
#   --save_folder ./weights/lr_1e3_mobinetv3/ \
