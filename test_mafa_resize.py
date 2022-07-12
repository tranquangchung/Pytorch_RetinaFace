from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import random
from utils.box_utils import matrix_iof
import math


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./wider_evaluate/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--path_test', type=str, help='Path Test')
args = parser.parse_args()
print(args)

def crop(image, boxes, labels, landm):
    height, width, _ = image.shape
    pad_image_flag = True
    h_new, w_new = 360, 640 

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        # short_side = min(width, height)
        # w = int(scale * short_side)
        # h = w
        w = int(width * scale)
        h = int(height * scale)

        if width <= w:
            l = 0
        else: # width > w
            l = random.randrange(width - w)
        if height <= h:
            t = 0
        else: # height > w
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

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
        ratio = math.ceil(max(h_roi/h_new, w_roi/w_new))
        h_roi_new = int(h_roi / ratio)
        w_roi_new = int(w_roi / ratio)
        image_t = cv2.resize(image_t, (w_roi_new, h_roi_new))
        scale_h, scale_w = float(h_roi_new / h_roi), float(w_roi_new / w_roi)
        boxes_t[:, 0::2] *= scale_w 
        boxes_t[:, 1::2] *= scale_h
        landms_t[:, 0::2] *= scale_w
        landms_t[:, 1::2] *= scale_h
        #######################

        # boxes_tmp = boxes_t
        # x1 = int(boxes_tmp[:, 0][0])
        # y1 = int(boxes_tmp[:, 1][0])
        # x2 = int(boxes_tmp[:, 2][0])
        # y2 = int(boxes_tmp[:, 3][0])
        # print(x1, y1, y2, y2)
        # image_t = cv2.rectangle(image_t, (x1, y1), (x2, y2), (0,255,0), 2)
        # for i in range(0, 10, 2):
        #     p1, p2 = int(landms_t[0][i]), int(landms_t[0][i+1])
        #     image_t = cv2.circle(image_t, (p1, p2), 1, (0, 255, 0), 2)
        # cv2.imwrite("abc.png", image_t)
        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, boxes, labels, landm, pad_image_flag

def pad_to_rectangle(image, rgb_mean, pad_image_flag):
    # if not pad_image_flag:
    #     return image
    # print(image.shape)
    try:
        height, width, _ = image.shape
        image_t = np.empty((360, 640, 3), dtype=image.dtype)
        image_t[:, :] = rgb_mean
        image_t[0:0 + height, 0:0 + width] = image
    except Exception as e:
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image_t = cv2.resize(image, (640, 360), interpolation=interp_method) # (w, h)
    return image_t

def resize_subtract_mean(image, rgb_mean):
    # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # interp_method = interp_methods[random.randrange(5)]
    # # image = cv2.resize(image, (640, 360), interpolation=interp_method) # (w, h)
    # image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)
    # return image

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndar0ray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(testset_folder, generator):
    # testing begin
    rgb_mean = (104, 117, 123)
    num_images = len(generator.keys())
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    all_detections = {}
    all_bounding_boxes = {}
    for i, (image_path, labels) in enumerate(generator.items()):
        img_name = image_path.split("/")[-1]
        img_path = str(testset_folder + image_path)
        img_raw = cv2.imread(img_path)
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        # img = np.float32(img_raw)
        bounding_boxes = np.zeros((0,4))
        landmarks = np.zeros((0,10))
        true_labels = np.zeros((0,1))
        for idx, t_label in enumerate(labels):
            label = t_label.split(" ")
            bounding_box = np.zeros((1, 4))
            landmark = np.zeros((1,10))
            true_label = np.zeros((1,1))
            # bbox
            bounding_box[0, 0] = label[0]  # x1
            bounding_box[0, 1] = label[1]  # y1
            bounding_box[0, 2] = int(label[0]) + int(label[2])  # x2
            bounding_box[0, 3] = int(label[1]) + int(label[3])  # y2
            # landmarks
            landmark[0, 0] = float(label[4])    # l0_x
            landmark[0, 1] = float(label[5])    # l0_y
            landmark[0, 2] = float(label[7])    # l1_x
            landmark[0, 3] = float(label[8])    # l1_y
            landmark[0, 4] = float(label[10])   # l2_x
            landmark[0, 5] = float(label[11])   # l2_y
            landmark[0, 6] = float(label[13])  # l3_x
            landmark[0, 7] = float(label[14])  # l3_y
            landmark[0, 8] = float(label[16])  # l4_x
            landmark[0, 9] = float(label[17])  # l4_y
            if (landmark[0, 0]<0):
                true_label[0, 0] = -1
            else:
                 true_label[0,0] = 1
            bounding_boxes = np.append(bounding_boxes, bounding_box, axis=0)
            landmarks = np.append(landmarks, landmark, axis=0)
            true_labels = np.append(true_labels, true_label, axis=0)
        img, bounding_boxes, true_labels, landmarks, pad_image_flag = crop(img, bounding_boxes, true_labels, landmarks)
        img = pad_to_rectangle(img, rgb_mean, pad_image_flag)
        img_show = img
        new_height, new_width, _ = img.shape
        # bounding_boxes[:, 0::2] /= new_width
        # bounding_boxes[:, 1::2] /= new_height

        all_bounding_boxes[img_name] = bounding_boxes
        ################
        # for b in bounding_boxes:
        #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        # cv2.imwrite("./results/abc1.png", img)
        ################
        # testing scale
        # target_size = 640
        # max_size = 800
        # im_shape = img.shape
        # im_size_min = np.min(im_shape[0:2])
        # im_size_max = np.max(im_shape[0:2])
        # resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        # if np.round(resize * im_size_max) > max_size:
        #     resize = float(max_size) / float(im_size_max)
        # if args.origin_size:
        #    resize = 1

        # if resize != 1:
        #     img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        # print(im_height, im_width)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        img = resize_subtract_mean(img, rgb_mean)
        # img -= (104, 117, 123)
        # img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        ######
        resize = 1
        boxes = boxes * scale / resize
        ######
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        # landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]


        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()
        box_detect = []
        for box in dets:
            tmp = [box[0], box[1], box[2], box[3], box[4]] # x1y1 x2y2
            box_detect.append(tmp)
        box_detect = np.array(box_detect)
        all_detections[img_name] = box_detect
        # --------------------------------------------------------------------

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_show, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_show, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_show, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_show, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_show, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_show, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            if not os.path.exists("./results/"):
                 os.makedirs("./results/")
            name = "./results/" + img_name
            cv2.imwrite(name, img_show)
    return all_detections, all_bounding_boxes

def _get_annotations(generator, bounding_boxes):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    num_classes = 1
    num_images = len(generator.keys())
    # all_annotations = [[None for i in range(num_classes] for j in range(num_images)]
    all_annotations = {}
    for key, value in generator.items():
        image_name = key.split("/")[-1]
        tmp = []
        boxes = bounding_boxes[image_name]
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            w = int(box[2])
            h = int(box[3])
            tmp1 = [x1, y1, int(x1+w), int(y1+h)] # x1y1 , x2y2
            tmp.append(tmp1)
        # print(tmp)
        tmp = np.array(tmp).astype(int)
        all_annotations[image_name] = tmp
    return all_annotations

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("pytorch_total_params", pytorch_total_params)

    # testing dataset
    testset_folder = args.dataset_folder

    with open(args.path_test) as f:
        lines = f.readlines()
        _fp_bbox_map = {}
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                name = line[1:].strip()
                _fp_bbox_map[name] = []
                continue
            _fp_bbox_map[name].append(line)
    all_detections, all_annotations = _get_detections(testset_folder, _fp_bbox_map) 
    # all_annotations = _get_annotations(_fp_bbox_map, all_bounding_boxes)

    average_precisions = {}
    iou_threshold = 0.5
    score_threshold = 0.05
    max_detections = 100

    # for label in range(generator.num_classes()):
    generator = _fp_bbox_map

    false_positives = np.zeros((0,))
    true_positives  = np.zeros((0,))
    scores          = np.zeros((0,))
    num_annotations = 0.0

    for key, _ in generator.items():
        image_name = key.split("/")[-1]
        detections           = all_detections[image_name]
        annotations          = all_annotations[image_name]
        num_annotations     += annotations.shape[0]
        detected_annotations = []

        for d in detections:
            scores = np.append(scores, d[4])

            if annotations.shape[0] == 0:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)
                continue
            overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap         = overlaps[0, assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives  = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)

    # sort by score
    indices         = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives  = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)

    # compute recall and precision
    recall    = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision  = _compute_ap(recall, precision)
    print('\nmAP:')
    print(average_precision)
