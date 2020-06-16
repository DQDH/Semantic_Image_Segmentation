import os,shutil

file_path='../voc12/train_aug_id3.txt'
src_dir='/home1/zqqHD/Smantic_Segmentation/AffinityNet/psa-master/result/dsrg-aff_vggtrain/out_rw10/'
dst_dir='/home1/zqqHD/Smantic_Segmentation/AffinityNet/psa-master/result/dsrg-aff_vggtrain/out_rw10_3/'

if not os.path.exists(file_path):
    print('label list not exist!')
if not os.path.exists(src_dir):
    print('src dir not exist!')
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
f=open(file_path)
labels=f.read()
labels=labels.split('\n')
f.close()
for label in labels:
    srcimage=os.path.join(src_dir,label+'.png')
    # dstimage=os.path.join(dst_dir,label+'.png')
    if not os.path.exists(srcimage):
        print(label, 'not exist')
    shutil.move(srcimage,dst_dir)
    # shutil.copyfile(srcimage, dstimage)

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile))

srcfile='/Users/xxx/git/project1/test.sh'
dstfile='/Users/xxx/tmp/tmp/1/test.sh'

mymovefile(srcfile,dstfile)
