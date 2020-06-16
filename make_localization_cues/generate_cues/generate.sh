python generate_cues.py --infer_list ../../list/train_aug.txt --network network.vgg16_cls --weights ../models/localization_cues/vgg_cls8.pth \
--out_cam ./results/out_cam/ --out_la_crf ./result/vggtrain1/out_la_crf/ --out_ha_crf ./result/vggtrain1/out_ha_crf/ --out_cam_pred ./results/out_cam/ \
--test_root /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/JPEGImages --pretrained_model ../../pretrained_model/poolnet//resnet50_caffe.pth \
--mode test --model ../models/sod_poolnet/final.pth --drfi_dir /home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/train_aug_val_DRFI --test_fold results --gpu 0 \
--picke_name ./results/pickle/localization_cues_mys >>time.txt
