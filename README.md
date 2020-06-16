基于超像素扩展和显著性目标扩展的关键词弱监督图像语义分割

step1：根据关键词标签训练多标签分类网络获得模型，以便后续获得类激活图
训练及测试脚本见  MySeg/make_localization_cues/get_localization_cues_model/路径下的run.sh

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

