# CUDA_VISIBLE_DEVICES=1 python test_widerface.py \
#   --trained_model ./weights/mobilenet0.25_Final.pth \
#   --network mobile0.25

CUDA_VISIBLE_DEVICES=1 python test_widerface.py \
  --trained_model ./weights/lr_1e3_resize_image_rgb_relu/mobilenet0.25_epoch_235.pth \
  --network mobile0.25 \
  --save_folder ./widerface_evaluate/widerface_txt_mobile_rgb_relu_235
