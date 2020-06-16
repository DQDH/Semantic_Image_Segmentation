import numpy as np
import cv2
import os
from scipy.ndimage import zoom
def generate_cam_img(cuenpy,label_src,fore_p,dict,id):
    label_index = np.where(label_src == 1)[0] + 1
    dict[id + '_labels'] = label_index
    label = np.zeros([cuenpy.shape[1], cuenpy.shape[2]])
    for i in label_index:
        label_i = cuenpy[i]
        label_i[label_i <= label_i.max() * fore_p] = 0
        label_i[label_i > label_i.max() * fore_p] = i
        label = label + label_i
    for l in np.unique(label):
        if l not in label_index:
            label[label == l] = 0
    return label
def get_drfi_bg(cue_img,drfi_dir,img_id):
    drfi_img = np.transpose(cv2.imread(os.path.join(drfi_dir, img_id + '.jpg')), [2, 0, 1])[0]
    drfi_img = zoom(drfi_img, (41.0 / drfi_img.shape[0], 41.0 / drfi_img.shape[1]), order=1)
    drfibg_index = np.where(drfi_img <= 9)
    cue_img[cue_img==0]=22
    cue_img[drfibg_index[0], drfibg_index[1]] = 0
    return cue_img
