#train cue model
#python train_cls.py --gpu 0 --session_name ./models/vgg_cls --train_list ../../list/train_aug.txt --lr 0.1 --max_epoches 30 --batch_size 32 --crop_size 448 --network network.vgg16_cls
model=1
while (($model<31))
do
    python3 infer_cls.py --infer_list ../../list/init_cue_val.txt --network network.vgg16_cls --weights ./models/vgg_cls${model}.pth --out_cam_pred /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/make_cue_model/model_val${model}/ --gpu 2
    echo $model
    model=`expr $model + 1`
done
