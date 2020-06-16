import sys
sys.path.append('../get_localization_cues_model/')
sys.path.append('../get_sod_model/PoolNet-master/')
from dataset.joint_dataset import get_loader
from joint_solver_test import Solver
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import tool.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
import csv
import pickle as cPickle
import copy
from tools.init_cues import generate_cam_img,get_drfi_bg
from tools.super_pixel import sup
from tools.sod import sod
from tools.make_cue_pickle import make_cue_pickle
import matplotlib.pyplot as plt

def _crf_with_alpha(cam_dict,orig_img, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ##cam parser
    parser.add_argument("--weights", default='../models/localization_cues/vgg_cls8.pth', type=str)  #
    parser.add_argument("--network", default="network.vgg16_cls", type=str)
    parser.add_argument("--infer_list", default="../../list/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--label_npy", default='../../list/cls_labels.npy', type=str)
    parser.add_argument("--voc12_root", default='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/', type=str)  #
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_cam", default='./result/vggtrain/out_cam/', type=str)  #
    parser.add_argument("--out_la_crf", default='./result/vggtrain/out_la_crf/', type=str)  #
    parser.add_argument("--out_ha_crf", default='./result/vggtrain/out_ha_crf/', type=str)  #
    parser.add_argument("--out_cam_pred", default='./result/vggtrain/out_cam_pred/', type=str)  #
    parser.add_argument("--gpu", default="1", type=str)

    #sod-poolnet parser
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--pretrained_model', type=str, default='../../pretrained_model/poolnet//resnet50_caffe.pth')
    parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')  # resnet or vgg
    parser.add_argument('--batch_size', type=int, default=1)  # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Testing settings
    parser.add_argument('--model', type=str, default='../models/sod_poolnet/final.pth')  # Snapshot
    parser.add_argument('--test_fold', type=str, default='./results')  # Test results saving folder
    parser.add_argument('--test_mode', type=int, default=1)  # 0->edge, 1->saliency
    parser.add_argument('--test_root', type=str, default='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/JPEGImages')
    parser.add_argument('--image_name', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    #inti_cue parameter
    parser.add_argument("--fore_p", default=0.5, type=float)
    parser.add_argument("--drfi_dir", default='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/train_aug_val_DRFI/', type=str)

    #SOD parameter
    parser.add_argument("--gt_prob", default=0.8, type=float)
    parser.add_argument("--picke_name", default="./localization_cues-sal", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(args.test_fold): os.mkdir(args.test_fold)

    #cam_model
    cam_model = getattr(importlib.import_module(args.network), 'Net')()
    cam_model.load_state_dict(torch.load(args.weights))
    cam_model.eval()
    cam_model.cuda()
    infer_dataset = tool.data.VOC12ClsDatasetMSF(args.infer_list, args.label_npy, voc12_root=args.voc12_root,
                                                 scales=(1, 0.5, 1.5, 2.0),
                                                 inter_transform=torchvision.transforms.Compose(
                                                     [np.asarray,
                                                      cam_model.normalize,
                                                      imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(cam_model, list(range(n_gpus)))
    print('cam model loaded!')

    # sod_poolnet_model
    sod_poolnet_model = Solver(None, None, args)
    print('sod_poolnet model loaded!')

    #make cues
    img_id={}
    with open('../../list/input_list-all.txt') as f:
        data=list(csv.reader(f))
    for line in data:
        line = line[0].split()
        img_id[line[0][0:-4]] = line[1]
    dict={}
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        print(iter)
        img_name = img_name[0];label = label[0]
        img_path = tool.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        h,w = orig_img.shape[:2]

        #cam
        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    cam = model_replicas[i % n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam, (41,41), mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()
        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)
        bg_score = [np.ones_like(norm_cam[0])*0]
        pred = np.concatenate((bg_score, norm_cam))
        pred[0]=np.max(pred,0)
        #cam_dict = {}
        #for i in range(20):
        #    if label[i] > 1e-5:
        #        cam_dict[i] = norm_cam[i]
        # if args.out_cam_pred is not None:
        #     bg_score = [np.ones_like(norm_cam[0]) * 0.2]
        #     pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
        #     scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))
        #use diffrent param CRF the cam
        # if args.out_la_crf is not None:
        #     crf_la = _crf_with_alpha(cam_dict, orig_img,args.low_alpha)
        #     np.save(os.path.join(args.out_la_crf, img_name + '.npy'), crf_la)
        #
        # if args.out_ha_crf is not None:
        #     crf_ha = _crf_with_alpha(cam_dict,orig_img, args.high_alpha)
        #     np.save(os.path.join(args.out_ha_crf, img_name + '.npy'), crf_ha)

        #use threshould segment the cam to get init_cue,and use drfi add backgroud,and ues superpixel to extend
        cam_img_fg=generate_cam_img(pred,label,args.fore_p,dict,img_id[img_name])
        init_cue=copy.deepcopy(cam_img_fg)
        init_cue=get_drfi_bg(init_cue,args.drfi_dir,img_name)
        sup_cue=sup(init_cue,orig_img)

        # sod_poolnet_img
        args.image_name=img_name
        test_loader = get_loader(args, mode='test')
        sod_poolnet_img = sod_poolnet_model.test(test_loader, test_mode=args.test_mode)
        # cv2.imwrite(os.path.join(args.test_fold, img_name + '.png'), sod_poolnet_img)

        #ues sod extend sup_cue
        res_cue=sod(img_name,cam_img_fg,sup_cue,sod_poolnet_img,orig_img,args.gt_prob)

        # #generate cue pickle
        # print(dict[img_id[img_name] + '_labels'])
        # dict[img_id[img_name] + '_cues'] = make_cue_pickle(res_cue,label)
        # output = open(args.picke_name + '.pickle', 'wb')
        # cPickle.dump(dict, output, protocol=2)
        # print(iter, img_name)