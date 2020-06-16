PASCAL_DIR=/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12
PASCAL_DATA_DIR=/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012
GPU=1
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


python ./tools/test-ms-f.py --model models/mycue8000-crf/model-f_mycue8000-crf_iter_16000.caffemodel --images ../list/val_id.txt --dir /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/ --output /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/voc12/mycue8000-crf/16000-nocrf1 --smooth False --gpu ${GPU}
