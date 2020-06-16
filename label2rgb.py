# -*-coding:utf-8-*-
import os
import numpy as np
import cv2
from PIL import Image,ImageDraw
# label_sd = 'E:/Semantic_Segmentation/DSRG-master/training/tools/'
# img_sd = 'E:/Semantic_Segmentation/DSRG-master/training/tools/1c'
label_sd = '/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/cue2png/mycue/valr'
img_sd = '/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/cue2png/mycue/valrc'
# label_sd = "E:\Semantic_Segmentation\ResultMap\OriginalMap\ori_cue_super_pixel_dss/"
# img_sd = "E:\Semantic_Segmentation\ResultMap\OriginalMap\ori_cue_super_pixel_dssc"
if not os.path.exists(label_sd):
    print("The label_sd path is not exist!")
    assert False
if not os.path.exists(img_sd):
    os.makedirs(img_sd)

palette=[]
for i in range(256):
    palette.extend((i,i,i))
# flatten: 2 dims to 1 dim , row is main
palette[:3*24]=np.array([[0,0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128],
                            [255, 0, 0],
                            [255, 255, 255],
                            [0, 0, 255]], dtype='uint8').flatten()

classes = {  'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4,
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8,
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12,
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'pottedplant'  : 16,
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tvmonitor'    : 20,
             'much'      : 21, 'less'       :22, 'diff'        :23}

for file in os.listdir(label_sd):
    label_pathname = os.path.join(label_sd,file)
    label_img = Image.open(label_pathname)
    label_img.putpalette(palette)
    label_img.save(os.path.join(img_sd, file))