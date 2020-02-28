"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

import random
import string

from craft import CRAFT
from collections import OrderedDict

canvas_size = 1280

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(result_folder,net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
        cv2.imwrite(result_folder+str(k)+".png",image[int(polys[k][0][1]):int(polys[k][0][1]+(polys[k][2][1]-polys[k][0][1])),\
            int(polys[k][0][0]):int(polys[k][0][0]+(polys[k][1][0]-polys[k][0][0]))])

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    # print(f"sl : {polys}")
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


#init
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
args = parser.parse_args()

# load net
net = CRAFT()     # initialize
trained_model = "weights/craft_mlt_25k.pth"
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4
cuda= True
refiner_model = 'weights/craft_refiner_CTW1500.pth'
#'enable link refiner'
refine = False

print('Loading weights from checkpoint (' + trained_model + ')')
if cuda:
    net.load_state_dict(copyStateDict(torch.load(trained_model)))
else:
    net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

if cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
net.eval()

# LinkRefiner
refine_net = None
if refine:
    from refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
    if cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

    refine_net.eval()
    args.poly = True

""" For test images in a folder """
image_list, _, _ = file_utils.get_files("data/")

def detect(image_path):
    result_folder = "result/"+str(time.time())+"/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    image = imgproc.loadImage(image_path)
    bboxes, polys, score_text = test_net(result_folder,net, image, text_threshold, link_threshold, low_text, cuda, args.poly, refine_net)
    return result_folder

detect("data/plat1.png")

