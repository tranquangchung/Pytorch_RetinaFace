CUDA_VISIBLE_DEVICES=0
python3 eval.py --csv_train ./data/train_wider.txt \
  --csv_val ./data1/val_wider_full.txt --csv_classes ./data/WIDER_classes.txt \
  --depth 18 --epochs 20 \
  --pretrained ./checkpoints/face_fan_18_new_40.pt \
  --model_name face_fan_18
