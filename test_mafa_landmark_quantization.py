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
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pdb


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./mafa_evaluate/', type=str, help='Dir to save txt results')
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
    b: (K, 4) ndarray of float
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


def _get_detections(generator):
    # testing begin
    num_images = len(generator.keys())
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    all_detections = {}
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 18), gridspec_kw={'height_ratios': [1, 1]})
    # ax1.set_xlabel("width")
    # ax1.set_ylabel("height")
    # ax2.set_xlabel("area")
    # areas = []
    for i, (image_path, _) in enumerate(generator.items()):
        img_name = image_path.split("/")[-1]
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        # target_size = 640
        # max_size = 800
        # im_shape = img.shape
        # im_size_min = np.min(im_shape[0:2])
        # im_size_max = np.max(im_shape[0:2])
        # resize = float(target_size) / float(im_size_min)
        # # prevent bigger axis from being more than max_size:
        # if np.round(resize * im_size_max) > max_size:
        #     resize = float(max_size) / float(im_size_max)
        # if args.origin_size:
        #     resize = 1
        resize = 1
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = ((img / 255) - 0.5) / 0.5
        # img -= (104, 117, 123)
        img = img / 255
        img = img.transpose(2, 0, 1)
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
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
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
            tmp = [
                    int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4], 
                    int(box[5]), int(box[6]), int(box[7]), int(box[8]),
                    int(box[9]), int(box[10]), int(box[11]), int(box[12]),
                    int(box[13]), int(box[14])
                ] # x1y1 x2y2
            box_detect.append(tmp)
            # width_box = int(box[2]*resize) - int(box[0]*resize)
            # height_box = int(box[3]*resize) - int(box[1]*resize)
            # area = width_box*height_box
            # areas.append(area)
            # ax1.scatter(width_box, height_box, c="blue")
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
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + img_name
            cv2.imwrite(name, img_raw)
        # if i > 100:
        #     # exit()
        #     break
    # ax2.hist(areas, bins=50)
    # plt.savefig("360_640_10.png")
    return all_detections

def _get_annotations(generator):
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
        for v in value:
            box = v.split(" ")
            x1 = int(box[0])
            y1 = int(box[1])
            w = int(box[2])
            h = int(box[3])
            tmp1 = [x1, y1, int(x1+w), int(y1+h)] # x1y1 , x2y2
            # left_eye = [int(float(box[4])), int(float(box[5]))]
            # right_eye = [int(float(box[7])), int(float(box[8]))]
            # nose = [int(float(box[10])), int(float(box[11]))]
            # leftmouth = [int(float(box[13])), int(float(box[14]))]
            # rightmouth = [int(float(box[16])), int(float(box[17]))]
            landmarks = [
                int(float(box[4])), int(float(box[5])), 
                int(float(box[7])), int(float(box[8])), 
                int(float(box[10])), int(float(box[11])), 
                int(float(box[13])), int(float(box[14])), 
                int(float(box[16])), int(float(box[17]))
            ]
            tmp1.extend(landmarks)
            tmp.append(tmp1)
        tmp = np.array(tmp).astype(int)
        all_annotations[image_name] = tmp
    return all_annotations

def caculate_error(landmark_a, landmark_d, norm):
    # left_e, right_e, nose, left_m, right_m = caculate_error(landmark_a_assigned, landmark_d)
    left_e = distance.euclidean(landmark_a[0:2], landmark_d[0:2])
    oks_left_e = left_e/norm
    right_e = distance.euclidean(landmark_a[2:4], landmark_d[2:4])
    oks_right_e = right_e/norm
    nose = distance.euclidean(landmark_a[4:6], landmark_d[4:6])
    oks_nose = nose/norm
    left_m = distance.euclidean(landmark_a[6:8], landmark_d[6:8])
    oks_left_m = left_m/norm
    right_m = distance.euclidean(landmark_a[8:10], landmark_d[8:10])
    oks_right_m = right_m/norm
    total_e = left_e + right_e + nose + left_m + right_m
    oks_total_e = oks_left_e + oks_right_e + oks_nose + oks_left_m + oks_right_m
    return oks_total_e, oks_left_e, oks_right_e, oks_nose, oks_left_m, oks_right_m
    # return total_e, left_e, right_e, nose, left_m, right_m

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    device = 'cpu'
    net = torch.jit.load(args.trained_model, map_location=device)
    print('Finished loading jit model!')
    # print(net)
    print("*"*20)
    print(net)
    cudnn.benchmark = True

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
    all_annotations = _get_annotations(_fp_bbox_map)
    all_detections = _get_detections(_fp_bbox_map) 

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
    total_e, left_e, right_e, nose, left_m, right_m = 0, 0, 0, 0, 0, 0
    count_face = 0
    for key, _ in generator.items():
        image_name = key.split("/")[-1]
        detections           = all_detections[image_name]
        annotations          = all_annotations[image_name]
        num_annotations     += annotations.shape[0]
        detected_annotations = []

        for d in detections:
            scores = np.append(scores, d[4])
            landmark_d = d[5:]
            landmark_a = annotations[:, 4:]
            if annotations.shape[0] == 0:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)
                continue
            overlaps            = compute_overlap(np.expand_dims(d[:4], axis=0), annotations[:, :4])
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap         = overlaps[0, assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives  = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
                landmark_a_assigned = landmark_a[assigned_annotation]
                # area = (annotations[assigned_annotation][:, 2] - annotations[assigned_annotation][:, 0]) * (annotations[assigned_annotation][:, 3] - annotations[assigned_annotation][:, 1])
                a = annotations[assigned_annotation][:, 2] - annotations[assigned_annotation][:, 0]
                b = annotations[assigned_annotation][:, 3] - annotations[assigned_annotation][:, 1]
                norm_c = np.sqrt(a**2 + b**2)
                total_e_tmp, left_e_tmp, right_e_tmp, nose_tmp, left_m_tmp, right_m_tmp = caculate_error(landmark_a_assigned.squeeze(), landmark_d, norm_c)
                total_e += total_e_tmp
                left_e += left_e_tmp
                right_e += right_e_tmp
                nose += nose_tmp
                left_m += left_m_tmp
                right_m += right_m_tmp
                count_face += 1
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
    print("Landmark Error: \n")
    print("True Positives: ", count_face)
    print("Total: {0}, Left Eye: {1}, Right Eye: {2}, Nose: {3}, Left Mouth: {4}, Right Mouth: {5}".format(
        total_e / count_face, left_e / count_face, right_e / count_face, 
        nose / count_face, left_m / count_face, right_m / count_face,
        ))
