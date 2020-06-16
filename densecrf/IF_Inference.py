"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""
import os
import numpy as np
import argparse
import pydensecrf.densecrf as dcrf
import copy
try:
    from cv2 import imread, imwrite
except ImportError:

    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


parser = argparse.ArgumentParser()
parser.add_argument("--im_dir", default='/home/zqq/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/JPEGImages/', type=str)
parser.add_argument("--anno_dir", default='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/DSRGOutput/', type=str)
parser.add_argument("--output_dir", default='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/DSRGOutput_crf/', type=str)
parser.add_argument("--gtprob", default=0.8, type=float)
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
#name=os.listdir(args.anno_dir)
#for img_name in name:
#    fn_im = os.path.join(args.im_dir, img_name[0:-4] + '.jpg')
#    fn_output=os.path.join(args.output_dir,img_name)
#    fn_anno=os.path.join(args.anno_dir, img_name)
f= open("../list/train_aug_id.txt")
name=f.readlines()
for img_name in name:
    img_name=img_name.split()
    img_name=img_name[0]
    fn_im = os.path.join(args.im_dir, img_name + '.jpg')
    fn_output=os.path.join(args.output_dir,img_name + '.png')
    fn_anno=os.path.join(args.anno_dir, img_name + '.png')
    print(img_name)
    img = imread(fn_im)
    anno = imread(fn_anno).astype(np.uint32)
    anno = anno[:,:,1]
    gt, labels = np.unique(anno, return_inverse=True)
    n_labels = len(set(labels.flat))
    NL = np.unique(labels)
    if n_labels==1:
        anno_n = imread(fn_anno)
        imwrite(fn_output, anno_n[:,:,1])
        continue

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=args.gtprob, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(121, 121), srgb=(5, 5, 5), rgbim=img,
                           compat=4,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)

    FMAP = copy.deepcopy(MAP)
    for i in range(len(NL)):
        ind=np.where(MAP==i)
        FMAP[ind] = gt[i]
    imwrite(fn_output, FMAP.reshape(img.shape[0],img.shape[1]))

