# -*- coding: UTF-8 -*-
import argparse
from pathlib import Path
import sys
import csv
from csv import writer
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages
from utils.general import check_img_size, non_max_suppression_face


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords
 
def detect(
    model,
    source,
    device,
    img_size,
    conf_thres,
    mode,
    model_name,
):
    # Load model
    #img_size = 640
    iou_thres = 0.5
    imgsz=(img_size, img_size)
    
    
    # Dataloader
    print('loading images', source)
    dataset = LoadImages(source, img_size=imgsz)
    bs = 1  # batch_size
    
    for path, im, im0s, vid_cap in dataset:
        
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
        else:
            orgimg = im.transpose(1, 2, 0)
        
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
        if len(pred[0]) == 0: 
            found = False 
        else: 
            found = True
        
        #If model didn't find a face on image then we save it to csv file
        if (found == False): 
            if mode == "train":
                file  = "fairface/"+ model_name+"/conf_thres="+ str(conf_thres) + "/train_undetected.csv"
                file_exists = os.path.isfile(file)
            else:
                file  = "fairface/" + model_name + "/conf_thres="+ str(conf_thres) + "/val_undetected.csv"
                file_exists = os.path.isfile(file)

            with open (file, 'a') as csvfile:
                headers = ['file']
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({'file': path})
                        


if __name__ == '__main__':
    mode = "val" #either "val"
    model_name = "RetinaFace" #name for folder
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/.pt', help='model.pt path(s)')

    if mode == "train":
        parser.add_argument('--source', type=str, default='fairface/train', help='source')  # file/folder
    else:
        parser.add_argument('--source', type=str, default='fairface/val', help='source')  # file/folder

    
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixesls)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence thr')
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.img_size, opt.conf_thres, mode, model_name)