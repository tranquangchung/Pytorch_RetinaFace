# CUDA_VISIBLE_DEVICES=1 python test_widerface.py \
#   --trained_model ./weights/mobilenet0.25_Final.pth \
#   --network mobile0.25

CUDA_VISIBLE_DEVICES=1 python test_widerface.py \
  --trained_model ./weights/lr_1e3_resize_image_norm_rgb/mobilenet0.25_Final.pth \
  --network mobile0.25 \
  --save_folder ./widerface_evaluate/widerface_txt_50_resize_image_norm_rgb_final
