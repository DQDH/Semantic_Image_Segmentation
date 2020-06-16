clc;clear;
Imgdir='/home/zqq/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/out_rw/';
Savedir='/home/zqq/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/out_rw_png/';
Name=dir([Imgdir '*.png']);
color=[[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]];
for n=1:length(Name)
    ImgPath=[Imgdir Name(n).name];
    Img=imread(ImgPath);
    [mm,nn]=size(Img);
    ColorMap=zeros(mm,nn,3);
    for i=1:mm
        for j=1:nn
            ColorMap(i,j,:)=color(Img(i,j)+1,:);
        end
    end
    imwrite(uint8(ColorMap),[Savedir Name(n).name])
end
