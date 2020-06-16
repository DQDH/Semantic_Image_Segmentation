import importlib
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import csv
import os

#imput image and model
pthfile = r'/home1/zqqHD/Smantic_Segmentation/AffinityNet/psa-master/model/p-model-my/vgg_cls16.pth'
img_root_path='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/train_aug/'
Img_name_list='/home1/zqqHD/Smantic_Segmentation/MySeg/list/train_id.txt'
Img_name_all=[]
with open(Img_name_list) as f:
    csv_reader=list(csv.reader(f))
for e in csv_reader:
    Img_name_all.append(e[0])
classes = {0: 'aeroplane',  1: 'bicycle',    2: 'bird',        3: 'boat',
           4: 'bottle',     5: 'bus',        6: 'car',         7: 'cat',
           8: 'chair',      9: 'cow',       10:'diningtable', 11: 'dog',
           12: 'horse',     13: 'motorbike', 14: 'person',     15: 'pottedplant',
           16: 'sheep',     17:'sofa',       18:'train',       19:'tvmonitor'
           }
#loading model
net = getattr(importlib.import_module('network.vgg16_cls'), 'Net')()
net.load_state_dict(torch.load(pthfile))
finalconv_name = 'fc7'
net.eval()
print(net)

# hook the feature extractor(get finalconv_name layer feature)
features_blobs = []
#input: the finalconv_name layer's input
#output: the finalconv_name layer's output
def hook_feature(module, input, output):
    print("hook input", input[0].shape)
    features_blobs.append(output.data.cpu().numpy())

# register finalconv_name layer, and append the output of specific layer in features_blobs
# when the net() be executed,the registered hook will be executed,it means executed finalconv_name layer
net._modules.get(finalconv_name).register_forward_hook(hook_feature)
print(net._modules)

# get the softmax weight
# convert params to list, array by weights bias,no params in pooling
params = list(net.parameters())
# extract params of softmax layer
weight_softmax = np.squeeze(params[-1].data.numpy())

# process data，resize to（224*224）and convert data type to tensor, finally normalize
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    normalize
])
for e in Img_name_all:
    img_pil = Image.open(os.path.join(img_root_path,e+'.jpg'))
    size_upsample = img_pil.size
    img_tensor = preprocess(img_pil)
    # convert image to Variable type
    img_variable = Variable(img_tensor.unsqueeze(0))
    #input image to network to get the predict score
    logit = net(img_variable)

    # use softmax to score
    h_x = F.softmax(logit, dim=1).data.squeeze()
    #sort the class predict score,output predict and its index in list
    probs, idx = h_x.sort(0, True)
    # convert data type
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction,top5
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # the function of creating CAM,complete the multiplication operation of weight and feature,finally,resize
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        # class_idx为预测分值较大的类别的数字表示的数组，一张图片中有N类物体则数组中N个元素
        for idx in class_idx:
            # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，故reshape了map的形状)
            # 把原来的相乘再相加转化为矩阵,w1*c1 + w2*c2+ .. -> (w1,w2..) * (c1,c2..)^T -> (w1,w2...)*((c11,c12,c13..),(c21,c22,c23..))
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            # 将feature_map的形状reshape回去
            cam = cam.reshape(h, w)
            # 归一化操作（最小的值为0，最大的为1）
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
            # 转换为图片的255的数据
            cam_img = np.uint8(255 * cam_img)
            # resize 图片尺寸与输入图片一致
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    # generate class activation mapping for the top1 prediction,output CAM image which size is same as imput image
    CAMs = returnCAM(features_blobs[0], weight_softmax, idx[:2])
    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])

    # concat image and CAM to show the localization result
    img = cv2.imread(os.path.join(img_root_path,e+'.jpg'))
    # create heatmap
    heatmap = cv2.applyColorMap(CAMs[1], cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/LocationMapTrain/'+e+'.jpg', result)