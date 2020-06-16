import numpy as np
import copy
import cv2
import operator
from scipy.ndimage import zoom
from skimage import measure
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def CRF(img,anno,gt_prob):
    gt, labels = np.unique(anno, return_inverse=True)
    gt=gt[1:]
    n_labels = len(set(labels.flat)) - 1
    NL = np.unique(labels)
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=True)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(5, 5, 5), rgbim=img,
                           compat=4,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    MAP = np.argmax(Q, axis=0)
    FMAP = copy.deepcopy(MAP)
    for i in range(len(NL) - 1):
        ind = np.where(MAP == i)
        FMAP[ind] = gt[i]
    index = np.where(FMAP == 21)
    FMAP[index] = 0
    return FMAP.reshape(img.shape[0],img.shape[1])
def sod(img_name,cam_img_fg,sup_cue_img,sod_img,orig_img,gt_prob):
    sup_cue_img_Nobg = copy.deepcopy(sup_cue_img)
    sup_cue_img_Nobg[sup_cue_img == 22] = 0
    dss_img = zoom(sod_img, (41.0 / sod_img.shape[0], 41.0 / sod_img.shape[1]), order=1)
    dss_img_TV = cv2.threshold(dss_img, 150, 255, cv2.THRESH_BINARY)
    dss_img_TV_labels = measure.label(dss_img_TV[1], connectivity=1)
    dss_c=np.delete(np.unique(dss_img_TV_labels),0)
    #out for fore obj
    out = np.zeros_like(sup_cue_img)
    for dss_c_i in dss_c:
        label_index_i=np.where(dss_img_TV_labels==dss_c_i)
        cue_c=np.unique(sup_cue_img_Nobg[label_index_i[0],label_index_i[1]])
        if len(cue_c)==2 and cue_c[0]==0:
            out[label_index_i[0],label_index_i[1]]=cue_c[1]
        elif len(cue_c) == 1 and cue_c[0] > 0:
            out[label_index_i[0], label_index_i[1]] = cue_c[0]
        elif (cue_c[0] > 0 and len(cue_c) > 1) or len(cue_c) > 2:
            out1 = 21*np.ones_like(out)
            out1[label_index_i[0], label_index_i[1]] = sup_cue_img_Nobg[label_index_i[0], label_index_i[1]]
            img = zoom(orig_img, (41.0 / orig_img.shape[0], 41.0 / orig_img.shape[1], 1), order=1)
            crf_img = CRF(img, out1,gt_prob)
            out[label_index_i[0], label_index_i[1]] = crf_img[label_index_i[0], label_index_i[1]]
    cue_img_TV = copy.deepcopy(cam_img_fg)
    cue_img_TV[cue_img_TV != 0] = 1
    cue_img_TV_labels = measure.label(cue_img_TV, connectivity=1)
    cue_img_TV_c = np.delete(np.unique(cue_img_TV_labels), 0)
    for cue_img_TV_c_i in cue_img_TV_c:
        label_index_i = np.where(cue_img_TV_labels == cue_img_TV_c_i)
        out_c = list(np.unique(out[label_index_i[0], label_index_i[1]]))
        if operator.eq(out_c, [0]):
            out[label_index_i[0], label_index_i[1]] = cam_img_fg[label_index_i[0], label_index_i[1]]
    # out for bg obj
    out[out==0]=22
    sup_bg_indedx=np.where(sup_cue_img==0)
    out[sup_bg_indedx[0],sup_bg_indedx[1]]=0
    return out
