import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
from PIL import Image
import os


def plot_single_scale(img_id,scale_lst, name_lst, size):
    #pylab.rcParams['figure.figsize'] = size, size/2
    #plt.figure()
    for i in range(0, len(scale_lst)):
        plt.imsave(os.path.join(out_root_dir,name_lst[i]+'/'+img_id+'.png'),scale_lst[i], cmap=cm.Greys_r)
        # s = plt.subplot(1,5,i+1)
        # s.set_xlabel(name_lst[i], fontsize=10)
        # s.set_xticklabels([])
        # s.set_yticklabels([])
        # s.yaxis.set_ticks_position('none')
        # s.xaxis.set_ticks_position('none')
    #plt.tight_layout()

# Make sure that caffe is on the python path:
caffe_root = '../../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
EPSILON = 1e-8
data_root = '/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/val'
out_root_dir='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/train_aug_val_DSS'
with open('./lists/val_id.txt') as f:
    test_lst = f.readlines()

#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
# choose which GPU you want to use
caffe.set_device(1)
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('deploy.prototxt', 'dss_model_released.caffemodel', caffe.TEST)

# load image
for img_id in test_lst:
    img_id=img_id.split()[0]
    img = Image.open(os.path.join(data_root,img_id+'.jpg'))
    img = np.array(img, dtype=np.uint8)
    im = np.array(img, dtype=np.float32)
    im = im[:, :, ::-1]
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = im.transpose((2, 0, 1))


    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    # run net and take argmax for prediction
    net.forward()
    out1 = net.blobs['sigmoid-dsn1'].data[0][0, :, :]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0, :, :]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0, :, :]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0, :, :]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0, :, :]
    out6 = net.blobs['sigmoid-dsn6'].data[0][0, :, :]
    fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]
    res = (out3 + out4 + out5 + fuse) / 4
    res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
    out_lst = [out1, out2, out3, out4, out5,out6, fuse, res]
    name_lst = ['SOG1', 'SOG2', 'SOG3', 'SOG4', 'SOG5','SOG6', 'Fuse', 'Result']
    plot_single_scale(img_id,out_lst, name_lst, 16)

    # out1.save(os.path.join(out_root_dir, 'out1/'+img_id+'.png'))
    # out2.save(os.path.join(out_root_dir, 'out2/'+img_id+'.png'))
    # out3.save(os.path.join(out_root_dir, 'out3/'+img_id+'.png'))
    # out4.save(os.path.join(out_root_dir, 'out4/'+img_id+'.png'))
    # out5.save(os.path.join(out_root_dir, 'out5/'+img_id+'.png'))
    # out6.save(os.path.join(out_root_dir, 'out6/'+img_id+'.png'))
    # fuse.save(os.path.join(out_root_dir, 'fuse/'+img_id+'.png'))
    # res.save(os.path.join(out_root_dir,'res/'+img_id+'.png'))
