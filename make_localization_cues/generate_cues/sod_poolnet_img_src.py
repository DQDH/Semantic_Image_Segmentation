import argparse
import os
import sys
sys.path.append('../get_sod_model/PoolNet-master/')
from dataset.joint_dataset import get_loader
from joint_solver import Solver

def main(config,image_name):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.image_name=image_name
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test(test_mode=config.test_mode)
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':
    
    vgg_path = '../../pretrained_model/poolnet/vgg16_20M.pth'
    resnet_path = '../../pretrained_model/poolnet//resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=3)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--train_list', type=str, default='')
    parser.add_argument('--train_edge_root', type=str, default='') # path for edge data
    parser.add_argument('--train_edge_list', type=str, default='') # list file for edge data

    # Testing settings
    parser.add_argument('--model', type=str, default=None) # Snapshot
    parser.add_argument('--test_fold', type=str, default=None) # Test results saving folder
    parser.add_argument('--test_mode', type=int, default=1) # 0->edge, 1->saliency
    parser.add_argument('--test_root', type=str, default=None)
    parser.add_argument('--image_name', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument("--gpu", default = "0", type = str)
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)
    main(config,config.image_name)

