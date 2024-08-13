%% Writen by Bian Chang 
% Project: Patch matching
% Date: 2019/10/21
% Edit 2019/11/18

%% 读入并处理MIHC图像
clear;close all;clc;
%  for flag = 15:16
%     if flag == 1
%         name = '[7367,43385]';
%     elseif flag == 2
%         name = '[7148,44883]';
%     elseif flag == 3
%         name = '[6778,47441]';
%     elseif flag == 4
%         name = '[6407,49798]';
%     elseif flag == 5
%         name = '[9117,42089]';
%     elseif flag == 6
%         name = '[8899,45052]';
%     elseif flag == 7
%         name = '[11053,43924]';
%     elseif flag == 8
%         name = '[10885,45506]';
%     elseif flag == 9
%         name = '[10952,46836]';
%     elseif flag == 10
%         name = '[10868,48536]';
%     elseif flag == 11
%         name = '[12669,43301]';
%     elseif flag == 12
%         name = '[12921,45153]';
%     elseif flag == 13
%         name = '[13157,47492]';
%     elseif flag == 14
%         name = '[14672,43991]';
%     elseif flag == 15
%         name = '[15227,46112]';
%     elseif flag == 16
%         name = '[16355,44715]';
%     end
for flag = 1:2
    if flag == 1
         name = '[14181,60117]';
    %elseif flag == 2
        %name = '[8899,45052]';
     end      
% MIHC_path = '/home/du/bc/AIimm/third_he_stain/DAPI2/';
% src = MIHC_path;
% srcsuffix = strcat('.jpg');
% files = dir(fullfile(src,strcat('*',srcsuffix)));
% for file_i = 1:length(files)
%     disp(file_i);
%     srcName = files(file_i).name; 
%     name = srcName(10:end-20)
%     nosurffixname = srcName(1:end - 4);
%    MIHC_patch_name = strcat(MIHC_path,srcName);  %读入DAPI彩色局部图像
 MIHC_patch_name = strcat('/home/du/bc/AIimm/fourth_he_stain/DAPI/','204464-230533_',...
     name,'_composite_image.tif');  %读入DAPI彩色局部图像
    MIHC_patch = imread(MIHC_patch_name) ;
    %MIHC_patch = gpuArray(MIHC_patch);
    mihc_gray = rgb2gray(MIHC_patch);
    mihc_gray_inv = 255 - mihc_gray;  %反色，使其与he的颜色更接近

%灰度标准化
originMIHC = im2double(mihc_gray_inv);
imgtemp = originMIHC./max(max(originMIHC));
img=255*(imgtemp-min(min(imgtemp)))/(1-min(min(imgtemp)));
img=uint8(img);
img_new = imadjust(img,[],[2.5/255;252.5/255]); %压缩1%的对比度做一个对比度增强
mihc = img_new;
%mihc = gpuArray(mihc);
%% 读入并处理HE图像
HE = strcat('/home/du/bc/AIimm/fourth_he_stain/crop_he/',...
     name,'.jpg'); 
rgb_HE = imread(HE);
count = 1;
temp_score = -10;
%rgb_HE = gpuArray(rgb_HE);
disp('candidate HE image read done')
HE_gray = rgb2gray(rgb_HE);
originHE = im2double(HE_gray);
imgtemp = originHE./max(max(originHE));
img=255*(imgtemp-min(min(imgtemp)))/(1-min(min(imgtemp)));
img=uint8(img);
img_new = imadjust(img,[],[2.5/255;252.5/255]); %压缩1%的对比度做一个对比度增强
HE = img_new;
%HE = gpuArray(HE);
%% 匹配过程

[m, n] = size(mihc);
[M, N] = size(HE);
% Mb = round(M/m)*m;
% Nb = round(N/n)*n;
Mb = M;
Nb = N;
Mb
Nb
%HE = imresize(HE,[Mb,Nb]);

for i = 1: 50 : Mb - m + 1
    for j = 1: 50: Nb - n +1
        count = count+1;
        MAP = HE(i:i+m-1,j:j+n-1);
        MAP = gpuArray(MAP);
        rgb_MAP = rgb_HE(i:i+m-1,j:j+n-1,:);
        score = corr2(mihc,MAP);
        if count >=2
            if score >= temp_score
                temp_score = score;
                temp_MAP = rgb_MAP;
            end
        end
    end
end
%temp_MAP = gather(temp_MAP);
result_txt_file = strcat('/home/du/bc/AIimm/fourth_he_stain/map_he/',name,'.txt');
score_result = fopen(result_txt_file,'w');
fprintf(score_result,'corr2:  %d\r\n',temp_score);
fclose(score_result);
figure(flag);
imshow(temp_MAP)
imwrite(temp_MAP,strcat('/home/du/bc/AIimm/fourth_he_stain/map_he/',name,'.jpg'),'jpg');
  
end

                
                
 
       
    
            
 