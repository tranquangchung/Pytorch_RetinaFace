CUDA_VISIBLE_DEVICE=0 python test_inthewild.py \
  --trained_model ./weights/lr_1e5_resize_640_640_transfer_learning/mobilenet0.25_Final.pth \
  --network mobile0.25 \
  --path_test xml_mafa/mafa3.txt \
  --save_image
