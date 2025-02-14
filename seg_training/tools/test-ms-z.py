import numpy as np
import pylab
import scipy.ndimage as nd
import png
import cv2, os
import os.path as osp
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
import findcaffe
import caffe
from optparse import OptionParser
import copy
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian,unary_from_labels
def CRF_label(img,anno):
    gt, labels = np.unique(anno, return_inverse=True)
    n_labels = len(set(labels.flat))
    NL = np.unique(labels)
    if n_labels==1:
        return anno
        #anno_n = imread(fn_anno)
        #imwrite(fn_output, anno_n[:,:,1])
        #continue

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=options.gtprob, zero_unsure=False)
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
    return FMAP.reshape(img.shape[0],img.shape[1])

def CRF(img,prob):
    prob=np.transpose(prob,[2,0,1])
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], prob.shape[0])
    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)
    unary = unary.astype(np.float32)
    d.setUnaryEnergy(unary)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    crf_result=np.array(Q).reshape((prob.shape[0], img.shape[0], img.shape[1]))
    result=np.argmax(crf_result, axis=0)
    return CRF_label(img,result)

parser = OptionParser()
parser.add_option("--images", dest="image_list", type='string',
                  help="Path to image")
parser.add_option("--dir", dest="data_dir", type='string',
                  help="Path to image")
parser.add_option("--model", dest="model", type='string',
                  help="Model weights")
parser.add_option("--net", dest="net", type='string',
                  help="Model weights", default='deploy.prototxt')
parser.add_option("--output", dest="output_dir", type='string',
                  help="Output png file name", default='')
parser.add_option("--smooth", dest="smooth", action='store_true',
                  help="Apply postprocessing")
parser.add_option('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
parser.add_option("--gtprob", dest="gtprob", type='float',
                  default=0.9)
options, _ = parser.parse_args()

smooth = options.smooth
image_list = options.image_list
data_dir = options.data_dir
model = options.model
output = options.output_dir
if output and (not osp.isdir(output)):
    os.makedirs(output)

mean_pixel = np.array([104.0, 117.0, 123.0])

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)


def write_to_png_file(im, f):
    #global palette
    #palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
    #w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
    #with open(f, "w") as ff:
    #    w.write(ff, im)
    cv2.imwrite(f, im)


def preprocess(image, size, mean_pixel=mean_pixel):

    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]),
                    size / float(image.shape[1]), 1.0),
                    order=1)

    image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return np.expand_dims(image, 0)


def predict_mask(image_file,img_id, smooth=True):

    im = pylab.imread(image_file)
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_all = 0
    for size in [241, 321, 401]:
        im_process = preprocess(im, size)
        net.blobs['images'].reshape(*im_process.shape)
        net.blobs['images'].data[...] = im_process
        net.forward()
        scores = np.transpose(net.blobs['fc8-SEC'].data[0], [1, 2, 0])
        scores = nd.zoom(scores, (d1 / scores.shape[0], d2 / scores.shape[1], 1.0), order=1)
        scores_all += scores

    scores_exp = np.exp(scores_all - np.max(scores_all, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)

    eps = 0.00001
    probs[probs < eps] = eps

    if smooth:
        result=CRF(im, probs)
    else:
        result = np.argmax(probs, axis=2)

    return result


if __name__ == "__main__":

    caffe.set_device(options.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(options.net, model, caffe.TEST)

    image_ids = [i.strip() for i in open(image_list) if not i.strip() == '']
    data_dir = osp.join(data_dir, 'JPEGImages')
    for index, img_id in enumerate(image_ids):
        print index, img_id
        image_file = osp.join(data_dir, img_id+'.jpg')
        mask = predict_mask(image_file,img_id, smooth=smooth)

        if output:
            save_path = osp.join(output, img_id+'.png')
            write_to_png_file(mask, save_path)
        else:
            fig = plt.figure()

            ax = fig.add_subplot('121')
            ax.imshow(pylab.imread(image_file))

            ax = fig.add_subplot('122')
            ax.matshow(mask, vmin=0, vmax=21, cmap=my_cmap)

            plt.show()
