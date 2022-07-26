from data.data_augment_1 import preproc1
from data import WiderFaceDetection 
training_dataset="/home/chungtran/Code/Image/Pytorch_Retinaface/data/widerface/train/label.txt"
rgb_mean = (104, 117, 123) # bgr order
img_dim = 640
widerface = WiderFaceDetection(training_dataset, preproc1(img_dim, rgb_mean))
# widerface.__getitem__(13)
for i in range(100000):
    print(i)
    widerface.__getitem__(i)
