import os
from matplotlib import pyplot as plt
import pickle as cPickle
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import zoom
origin_image_dir='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/JPEGImages'
Out_dir='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/cue2png/mycue/bg21/'
Pickle_path='/home1/zqqHD/Smantic_Segmentation/MySeg/make_localization_cues/generate_cues/results/pickle/localization_cues_my.pickle'
if not os.path.exists(Out_dir):
    os.makedirs(Out_dir)
data_file = cPickle.load(open(Pickle_path,'rb'))
f= open("../../../list/train_aug_id.txt")
image_id_All=f.readlines()
for i in range(len(image_id_All)):
    image_id=image_id_All[i].split()[0]
    cue = np.zeros([21, 41, 41])
    label_id = data_file['%i_labels' % i]
    cues_i = data_file['%i_cues' % i]
    cue[cues_i[0], cues_i[1], cues_i[2]] = 1.0
    label_id=np.append(0,label_id)
    origin_image_path = os.path.join(origin_image_dir, image_id+'.jpg')
    orig_img = np.asarray(Image.open(origin_image_path))
    h, w = orig_img.shape[:2]
    out = np.zeros([h, w])
    for label in label_id:
        N_label = cue[label, :, :]
        if label==0:
            N_label[N_label == 1] = 22
            N_label = cv2.resize(N_label, (w, h), interpolation=cv2.INTER_LINEAR)
            N_label[N_label != 22] = 0
        else:
            N_label[N_label == 1] =label
            N_label = cv2.resize(N_label, (w, h), interpolation=cv2.INTER_LINEAR)
            N_label[N_label != label] = 0
        out = out + N_label
    out[out>22]=0
    out[out==0]=21
    out[out==22]=0
    cv2.imwrite(os.path.join(Out_dir,image_id+'.png'), out)

