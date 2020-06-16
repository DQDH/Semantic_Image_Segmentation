% addpath(genpath('.'));
% 
% image_name = './data/2007_000032.jpg';
% image = imread( image_name );
% 
% para = makeDefaultParameters;
% 
% % acclerate using the parallel computing
% % matlabpool
% 
% t = tic;
% smap = drfiGetSaliencyMap( image, para );
% time_cost = toc(t);
% fprintf( 'time cost for saliency computation using DRFI approach: %.3f\n', time_cost );
% 
% subplot('121');
% imshow(image);
% subplot('122');
% imshow(smap);

addpath(genpath('.'));
ImgDir='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/val/';
ImgAll=dir([ImgDir,'*.jpg']);
SaveDir='/home1/zqqHD/DataSet/Pascal/VOCdevkit/VOC2012/val_DRFI/';
for i=1401:1449
    image_name=[ImgDir ImgAll(i).name];
    image = imread( image_name );

    para = makeDefaultParameters;
    t = tic;
    smap = drfiGetSaliencyMap( image, para );
    time_cost = toc(t);
    fprintf( 'time cost for saliency computation using DRFI approach: %.3f\n', time_cost );
%     subplot('121');
%     imshow(image);
%     subplot('122');
%     imshow(smap);
    imwrite(smap,[SaveDir ImgAll(i).name])
end
