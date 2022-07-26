import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof
import pdb
import math

h_new, w_new = 360, 640 
# h_new, w_new = 480, 640 
# h_new, w_new = 640, 640 

def _crop(image, boxes, labels, landm, img_dim=None):
    height, width, _ = image.shape
    pad_image_flag = True
    # h_new, w_new = 480, 640 

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        ########################################
        # PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        # scale = random.choice(PRE_SCALES)
        # # short_side = min(width, height)
        # # w = int(scale * short_side)
        # # h = w
        # w = int(width * scale)
        # h = int(height * scale)

        # if width <= w:
        #     l = 0
        # else: # width > w
        #     l = random.randrange(width - w)
        # if height <= h:
        #     t = 0
        # else: # height > w
        #     t = random.randrange(height - h)
        # roi = np.array((l, t, l + w, t + h))
        ########################################
        if height <= h_new and width <= w_new:
            l = 0
            t = 0
        elif height >= h_new and width >= w_new:
            if (width - w_new) > 0:
                l = random.randrange(width - w_new)
            else: # = 0
                l = 0
            if (height - h_new) > 0:
                t = random.randrange(height - h_new)
            else:
                t = 0
        elif height >= h_new and width <= w_new:
            if (height - h_new) > 0:
                t = random.randrange(height - h_new)
            else:
                t = 0
            l = 0
        elif height <= h_new and width >= w_new:
            t = 0
            if (width - w_new) > 0:
                l = random.randrange(width - w_new)
            else: # = 0
                l = 0
        roi = np.array((l, t, l + w_new, t + h_new))
        ########################################

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        h, w, _ = image_t.shape
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t = landms_t.reshape([-1, 10])


	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * w_new
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * h_new
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False
        ##### padding & resize here #####
        h_roi, w_roi, _ = image_t.shape 
        # ratio = math.ceil(max(h_roi/h_new, w_roi/w_new))
        ratio = min(h_new/h_roi, w_new/w_roi)
        h_roi_new = int(h_roi * ratio)
        w_roi_new = int(w_roi * ratio)
        image_t = cv2.resize(image_t, (w_roi_new, h_roi_new))

        scale_h, scale_w = float(h_roi_new / h_roi), float(w_roi_new / w_roi)
        boxes_t[:, 0::2] *= scale_w 
        boxes_t[:, 1::2] *= scale_h
        landms_t[:, 0::2] *= scale_w
        landms_t[:, 1::2] *= scale_h
        #######################

        # boxes_tmp = boxes_t
        # for i in range(boxes_tmp.shape[0]):
        #     x1 = int(boxes_tmp[:, 0][i])
        #     y1 = int(boxes_tmp[:, 1][i])
        #     x2 = int(boxes_tmp[:, 2][i])
        #     y2 = int(boxes_tmp[:, 3][i])
        #     image_t = cv2.rectangle(image_t, (x1, y1), (x2, y2), (0,255,0), 2)
        # for j in range(landms_t.shape[0]):
        #     for i in range(0, 10, 2):
        #         p1, p2 = int(landms_t[j][i]), int(landms_t[j][i+1])
        #         image_t = cv2.circle(image_t, (p1, p2), 1, (0, 255, 0), 2)
        return image_t, boxes_t, labels_t, landms_t, pad_image_flag

    ratio = min(h_new/height, w_new/width)
    h_roi_new = int(height * ratio)
    w_roi_new = int(width * ratio)
    image = cv2.resize(image, (w_roi_new, h_roi_new))

    scale_h, scale_w = float(h_roi_new / height), float(w_roi_new / width)
    boxes[:, 0::2] *= scale_w 
    boxes[:, 1::2] *= scale_h
    landm[:, 0::2] *= scale_w
    landm[:, 1::2] *= scale_h
    return image, boxes, labels, landm, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


# def _pad_to_square(image, rgb_mean, pad_image_flag):
#     if not pad_image_flag:
#         return image
#     height, width, _ = image.shape
#     long_side = max(width, height)
#     image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
#     image_t[:, :] = rgb_mean
#     image_t[0:0 + height, 0:0 + width] = image
#     return image_t

def _pad_to_rectangle(image, rgb_mean, pad_image_flag):
    # if not pad_image_flag:
    #     return image
    # print(image.shape)
    try:
        height, width, _ = image.shape
        image_t = np.empty((h_new, w_new, 3), dtype=image.dtype)
        # image_t = np.empty((480, 640, 3), dtype=image.dtype)
        image_t[:, :] = rgb_mean
        image_t[0:0 + height, 0:0 + width] = image
    except Exception as e:
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image_t = cv2.resize(image, (w_new, h_new), interpolation=interp_method) # (w, h)
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # interp_method = interp_methods[random.randrange(5)]
    # # image = cv2.resize(image, (640, 360), interpolation=interp_method) # (w, h)
    # image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    # image -= rgb_mean
    return image.transpose(2, 0, 1)
    # return image

def visualize(image_t, boxes_t, labels_t, landm_t, pad_image_flag):
    for i in range(boxes_t.shape[0]):
        x1 = int(boxes_t[:, 0][i])
        y1 = int(boxes_t[:, 1][i])
        x2 = int(boxes_t[:, 2][i])
        y2 = int(boxes_t[:, 3][i])
        image_t = cv2.rectangle(image_t, (x1, y1), (x2, y2), (0,255,0), 2)
    for j in range(landm_t.shape[0]):
        for i in range(0, 10, 2):
            p1, p2 = int(landm_t[j][i]), int(landm_t[j][i+1])
            image_t = cv2.circle(image_t, (p1, p2), 1, (0, 255, 0), 2)
    return image_t

class preproc1(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.index = 0

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        
        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()
        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
        # image_t = visualize(image_t, boxes_t, labels_t, landm_t, pad_image_flag)
        image_t = _distort(image_t)
        image_t = _pad_to_rectangle(image_t, self.rgb_means, pad_image_flag)
        height, width, _ = image_t.shape
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        #################
        # cv2.imwrite(f"./test_img/{self.index}.png", image_t)
        # self.index += 1
        #################
        # print(image_t.shape)
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))
        return image_t, targets_t
