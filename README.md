基于超像素扩展和显著性目标扩展的关键词弱监督图像语义分割
=====
环境配置：
-----
AffinityNet环境配置:<br>
CUDA=9.0<br>
1.创建虚拟环境：conda create -n Affinity python=3.6<br>
2.激活虚拟环境：source activate Affinity<br>
3.安装pytorch1.1.0：conda install pytorch torchvision cudatoolkit=9.0 -c pytorch<br>
4.安装caffe：conda install caffe-gpu<br>
5.降低numpy版本:conda install numpy=1.16.2<br>
6.安装pydensecrf：pip install pydensecrf<br>

PoolNet测试时，使用和AffinityNet相同的环境<br>
DSS-master采用的环境为dsrg环境，调用环境变量export PYTHONPATH=/home/zqq/caffe/python:$PYTHONPATH（原生caffe）<br>
MySeg工程中的make_localization_cues所使用的环境为Affinity<br>

数据路径:
-----
目录：/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012<br>
ImageSets<br>
JPEGImages<br>
SegmentationClass<br>
SegmentationClassAug_strong<br>
train_aug_val_DRFI<br>
voc12/JPEGImages<br>

算法流程：
-----
step1：根据关键词标签训练多标签分类网络获得模型，以便后续获得类激活图<br>
训练及测试脚本见  MySeg/make_localization_cues/get_localization_cues_model/路径下的run.sh<br>

step2：多类激活图获得的定位标签进行超像素扩展和显著性目标扩展
显著性目标检测方法有drfi、dss和poolnet，每个方法的具体工程代码见  MySeg/make_localization_cues/get_sod_model
扩展处理运行的脚本见 MySeg/make_localization_cues/generate_cues/下的generate.sh

step3:根据得到的标签训练语义分割模型
训练及测试脚本见  MySeg/seg_training下的run.sh

srep4:交并比计算
代码 MySeg/Test_mIoU/ValLabelEvalSegResults.m

tips:
代码中使用的drfi检测结果需要提前对训练样本进行处理，并保存结果
使用DSS显著性目标检测方法时，也需要提前对训练样本进行处理，并保存结果，并修改代码中显著性目标扩展的部分代码
使用Poolnet方法扩展时，直接运行脚本即可，无需提前处理样本。



路径：Semantic_Image_Segmentation/make_localization_cues/generate_cues/results/pickle/

[生成的pickle种子](https://pan.baidu.com/s/1Lpv_tFkc9VUsWzIvW7hxpg)提取码：038a

路径：Semantic_Image_Segmentation/make_localization_cues/get_localization_cues_model/models/vgg_cls8.pth

[多标签分类模型](https://pan.baidu.com/s/1ZfKZGBoS5iML6T6ZRBirfg)提取码：6p7z

路径：Semantic_Image_Segmentation/make_localization_cues/get_sod_model/drfi_matlab-master/model/

[drfi模型](https://pan.baidu.com/s/1wbgBaZ8cCWcYZGVQBMc7EA)提取码：wn9d 

路径：Semantic_Image_Segmentation/make_localization_cues/get_sod_model/DSS-master/

[DSS_Caffe_model_released](https://pan.baidu.com/s/1NVVpH-mZJwFvHRpI3hmT-g)提取码：rtib

[DSS_Caffe_model_best](https://pan.baidu.com/s/1mkdf1KzKi5EzEVFc9LNu1A)提取码：p4k5

[DSS_pytorch_model](https://pan.baidu.com/s/1O3XzTeEiwQXiInbV4U9Bqw)提取码：170k 

[DSS_pytorch_pretraind_model](https://pan.baidu.com/s/1IHySM0dnqgNwYyY8rbs-nA)提取码：qkc7

路径：Semantic_Image_Segmentation/make_localization_cues/get_sod_model/PoolNet-master/results/models/

[PoolNet_model](https://pan.baidu.com/s/1H9L9Bcji7iPts7t4nxhyxA)提取码：lwz2


路径：Semantic_Image_Segmentation/seg_training/models/

[segmodel](https://pan.baidu.com/s/1G5VMPB0Zw9Twyvt8VorS6A)提取码：rmmu

[segmodel](https://pan.baidu.com/s/1Dp-6nOh-TQ8u6TPvnXrzxw )提取码：5c5z

预训练模型：

路径：Semantic_Image_Segmentation/pretrained_model/seg/vgg16_20M_mc.caffemodel

[DSRG分割网络预训练模型](https://pan.baidu.com/s/1UvgHgA-9XgAg7UhUEoXKcg)提取码：0zxp

路径：Semantic_Image_Segmentation/pretrained_model/localization_cues/

[多标签分类网络预训练模型](https://pan.baidu.com/s/1-UpBZGs5Elki-W8P_vkpAA)提取码：gjjx

路径：Semantic_Image_Segmentation/pretrained_model/poolnet/

[PoolNet预训练模型](https://pan.baidu.com/s/1ScxdQh9G4bftY1BWu0Wnkg)提取码：ljq6 

[PoolNet预训练模型](https://pan.baidu.com/s/1gTxp2TO72lfIRKL-ieH3-Q)提取码：98id

