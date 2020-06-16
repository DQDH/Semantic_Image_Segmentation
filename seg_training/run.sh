PASCAL_DIR=/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12
PASCAL_DATA_DIR=/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012
GPU=3
#!/bin/bash
LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log
LOG0=log/train0-`date +%Y-%m-%d-%H-%M-%S`.log
# train step 1 (DSRG training)
#python ./tools/train.py --solver solver-s.prototxt --weights ../pretrained_model/seg/vgg16_20M_mc.caffemodel --gpu ${GPU} 2>&1 | tee $LOG
#python ./tools/train.py --solver solver-s.prototxt --weights ../pretrained_model/seg/vgg16_20M_mc.caffemodel --gpu ${GPU} 2>&1 | tee $LOG
#python ./tools/test-ms.py --model models/model-s-paper_iter_2000.caffemodel --images ../list/val_id.txt --dir /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012 --output /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/Myseg/2000 --gpu ${GPU} --smooth true

# train step 2 (retrain)
#python ../../tools/train.py --solver solver-f.prototxt --weights models/paper/model-s_iter_8000.caffemodel --gpu ${GPU}
#python ../../tools/test-ms-f.py --model models/model-f_iter_20000.caffemodel --images list/val_id.txt --dir /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/ --output /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/DSRG_final_output1 --smooth true --gpu ${GPU}

cd /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12
rm DSRGOutput
ln -s /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/mycue/3000_train_aug DSRGOutput
cd /home/zqq/zqqHD/Smantic_Segmentation/MySeg/seg_training
python ./tools/train.py --solver solver-f.prototxt --weights models/mycue/model-s-myseg_iter_3000.caffemodel --gpu ${GPU} 2>&1 | tee $LOG0
cd /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12
rm DSRGOutput
ln -s /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/DSRG/8000_train_aug-crf9 DSRGOutput
cd /home/zqq/zqqHD/Smantic_Segmentation/MySeg/seg_training
python ./tools/train.py --solver solver-f.prototxt --weights models/DSRG/model-s-DSRG_iter_8000.caffemodel --gpu ${GPU} 2>&1 | tee $LOG
cd /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12
rm DSRGOutput
ln -s /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/mycue/8000_train_aug-crf9 DSRGOutput
cd /home/zqq/zqqHD/Smantic_Segmentation/MySeg/seg_training
python ./tools/train.py --solver solver-f.prototxt --weights models/mycue/model-s-myseg_iter_8000.caffemodel --gpu ${GPU} 2>&1 | tee $LOG0
