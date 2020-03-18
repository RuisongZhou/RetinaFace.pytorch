from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import *
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import math
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_folder', default='./FDDB_Evaluation/fddb_evaluation/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='FDDB', type=str, choices=['FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--img_size', default=128.0, type=float)
args = parser.parse_args()


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


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    elif args.network == "MobileNetV3":
        cfg = cfg_mnetv3
    elif args.network == "GhostNet":
        cfg = cfg_ghostnet
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    # save file
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # testing dataset
    testset_folder = os.path.join('data', args.dataset, 'images/')
    testset_list = os.path.join('data', args.dataset, 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing scale
    resize = 0.5
    pbar = tqdm(range(num_images))
    pbar.set_description('Predicting ... ')
    # testing begin
    imageset = './FDDB_Evaluation/ground_truth'
    for j in range(10):
        annotationfilepath = imageset + "/FDDB-fold-%0*d.txt" % (2, j + 1)
        annotationfile = open(annotationfilepath)
        while True:
            filename = annotationfile.readline()[:-1] + ".jpg"
            if not os.path.exists(os.path.join(testset_folder, filename)):
                break
            img_raw = cv2.imread(os.path.join(testset_folder, filename), cv2.IMREAD_COLOR)
            img = np.float32(img_raw)
            im_size_max = np.max(img.shape[0:2])
            resize = args.img_size / float(im_size_max)
            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            loc, conf, landms = net(img)  # forward pass

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
            # order = scores.argsort()[::-1][:args.top_k]
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)

            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            # dets = dets[:args.keep_top_k, :]
            # landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # save dets
            if args.dataset == "FDDB":
                event_name = str(j + 1)
                if not os.path.exists(os.path.join(args.save_folder, event_name)):
                    os.makedirs(os.path.join(args.save_folder, event_name))

                fout = open(os.path.join(args.save_folder, event_name, filename.replace('/', '_')[:-3] + 'txt'),
                            'w')
                fout.write(filename.replace('/', '_')[:-4] + '\n')
                fout.write(str(dets.shape[0]) + '\n')
                for i in range(dets.shape[0]):
                    bbox = dets[i, :]
                    fout.write('%d %d %d %d %.03f' % (
                        math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]),
                        math.ceil(bbox[3] - bbox[1]),
                        bbox[4] if bbox[4] <= 1 else 1) + '\n')
                fout.close()


            # show image
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
                name = "./results/" + filename + ".jpg"
                cv2.imwrite(name, img_raw)

            pbar.update()