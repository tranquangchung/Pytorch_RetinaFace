# cuda_visible_devices=1 python test_widerface.py \
#   --trained_model ./weights/lr_1e5_resize_360_640_rgb_norm_01_transfer_learning/mobilenet0.25_final.pth \
#   --save_folder ./widerface_evaluate/widerface_txt_lr_1e5_resize_360_640_rgb_norm_01_transfer_learning \
#   --network mobile0.25 \
#   # --save_image

# CUDA_VISIBLE_DEVICES=0 python test_widerface.py \
#   --trained_model ./weights/mobilenet0.25_epoch_240.pth \
#   --network mobile0.25 \
#   --save_folder ./widerface_evaluate/widerface_txt_quantization \
#   # --save_image

# CUDA_VISIBLE_DEVICES=0 python test_widerface_quantization.py \
#   --trained_model ./weights/lr_1e3_resize_360_640_RGB_NORM_01_transfer_learning_quantization_L2/mobilenet0.25_Final_quantized_jit.pth \
#   --network mobile0.25 \
#   --save_folder ./widerface_evaluate/lr_1e3_resize_360_640_RGB_NORM_01_transfer_learning_quantization_L2/ \
#   # --save_image

# cuda_visible_devices=1 python test_widerface.py \
#   --trained_model ./weights/lr_1e3_mobinetv3/mobilenetv3_Final.pth \
#   --save_folder ./widerface_evaluate/widerface_txt_mobinetv3 \
#   --network mobinetv3 \
  # --save_image

CUDA_VISIBLE_DEVICES=0 python test_widerface_quantization.py \
  --trained_model ./weights/lr_1e5_resize_360_640_RGB_NORM_01_transfer_learning_quantization_L2/mobilenet0.25_Final_quantized_jit.pth \
  --network mobile0.25 \
  --save_folder ./widerface_evaluate/lr_1e5_resize_360_640_RGB_NORM_01_transfer_learning_quantization_L2 \
  # --save_image
