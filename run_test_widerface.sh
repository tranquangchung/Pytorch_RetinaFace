# CUDA_VISIBLE_DEVICES=1 python test_widerface.py \
#   --trained_model ./weights/mobilenet0.25_Final.pth \
#   --network mobile0.25

# CUDA_VISIBLE_DEVICES=0 python test_widerface.py \
#   --trained_model ./weights/mobilenet0.25_epoch_240.pth \
#   --network mobile0.25 \
#   --save_folder ./widerface_evaluate/widerface_txt_quantization \
#   # --save_image

CUDA_VISIBLE_DEVICES=0 python test_widerface_quantization.py \
  --trained_model ./weights/lr_1e3_resize_image_rgb_relu_conv_quantization/mobilenet0.25_Final_quantized_jit.pth \
  --network mobile0.25 \
  --save_folder ./widerface_evaluate/widerface_txt_quantization_relu_conv/ \
  --save_image
