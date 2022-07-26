# CUDA_VISIBLE_DEVICE=1 python test_mafa.py \
#   --trained_model ./weights/lr_1e3_resize_image_norm_rgb/mobilenet0.25_Final.pth \
#   --network mobile0.25 \
#   --path_test xml_mafa/mafa3.txt \
#   --save_image

# python test_mafa.py \
#   --trained_model ./weights/finetune/Resnet50_Final.pth \
#   --network resnet50 \
#   --path_test /mnt/disk1/chungtq/ComputerVision/FaceDetection/Pytorch_Retinaface/xml_mafa/mafa3.txt
# #   --save_image

# CUDA_VISIBLE_DEVICE=1 python test_mafa_landmark.py \
#   --trained_model ./weights/lr_1e3_resize_image_rgb_relu_conv/mobilenet0.25_epoch_170.pth \
#   --network mobile0.25 \
#   --path_test xml_mafa/mafa3.txt \
  # --save_image

# CUDA_VISIBLE_DEVICE=1 python test_mafa.py \
#   --trained_model ./weights/lr_1e3_resize_image_rgb_relu_conv/mobilenet0.25_epoch_170.pth \
#   --network mobile0.25 \
#   --path_test xml_mafa/mafa3.txt \
  # --save_image

CUDA_VISIBLE_DEVICE=1 python test_mafa_resize.py \
  --trained_model ./weights/lr_1e3_resize/lr_1e3_resizemobilenet0.25_epoch_245.pth \
  --network mobile0.25 \
  --path_test ./weights/Thoa_Resize/wider_test_medium.txt \
  # --save_image
