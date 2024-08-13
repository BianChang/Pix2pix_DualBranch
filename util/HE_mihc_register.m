% Written by Bian Chang
% Project: AI-IMM
% Function:
% Date: 2019/10/21
% Edit: 2020/1/5

clear; close all; clc;

% Define paths
HE_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE';
hema_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_hema\uint8';
dapi_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\RGB_3channels\normalized_channel_1';
bcl2_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\RGB_3channels\normalized_channel_5';
pax5_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\RGB_3channels\normalized_channel_9';

he_output_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE\Registered_HE';
dapi_output_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_dapi';
bcl2_output_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_bcl2';
pax5_output_path = 'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_pax5';

% Create output directories if they don't exist
disp('Creating output directories if they do not exist');
if ~exist(he_output_path, 'dir')
    mkdir(he_output_path);
end
if ~exist(dapi_output_path, 'dir')
    mkdir(dapi_output_path);
end
if ~exist(bcl2_output_path, 'dir')
    mkdir(bcl2_output_path);
end
if ~exist(pax5_output_path, 'dir')
    mkdir(pax5_output_path);
end

% Read and process each file
src = HE_path;
srcsuffix = strcat('.tif');
files = dir(fullfile(src, strcat('*', srcsuffix)));

for file_i = 1:length(files)
    disp(['Processing file ', num2str(file_i), ' of ', num2str(length(files))]);
    srcName = files(file_i).name;
    nosurffixname = srcName(4:end - 4);
    name = nosurffixname;

    % Read the images
    disp('Reading images');
    HE = imread(fullfile(HE_path, srcName));
    hema = imread(fullfile(hema_path, strcat('hema_', name, '.tif')));
    dapi = imread(fullfile(dapi_path, strcat('mIHC_', name, '_channel_1.tif')));
    bcl2 = imread(fullfile(bcl2_path, strcat('mIHC_', name, '_channel_5.tif')));
    pax5 = imread(fullfile(pax5_path, strcat('mIHC_', name, '_channel_9.tif')));

    fixed = hema;   
    moving = dapi; 
    moving2 = bcl2;
    moving3 = pax5;

    %% Rigid Registration
    disp('Performing rigid registration');
    [optimizer, metric] = imregconfig('multimodal');

    optimizer.MaximumIterations = 300;
    optimizer.InitialRadius = optimizer.InitialRadius / 7;

    disp('Calculating transformation matrix');
    tformSimilarity = imregtform(moving, fixed, 'similarity', optimizer, metric);
    disp('Transformation matrix calculated');

    if (~isSimilarity(tformSimilarity))
        tformSimilarity.T(1, :) = [1, 0, 0];
        tformSimilarity.T(2, :) = [0, 1, 0];
    end

    % Apply transformation to DAPI, BCL2, and PAX5
    disp('Applying transformation to DAPI, BCL2, and PAX5 images');
    movingRegisteredAffineWithIC = imwarp(moving, tformSimilarity, 'OutputView', imref2d(size(fixed)));
    movingRegisteredAffineWithIC2 = imwarp(moving2, tformSimilarity, 'OutputView', imref2d(size(fixed)));
    movingRegisteredAffineWithIC3 = imwarp(moving3, tformSimilarity, 'OutputView', imref2d(size(fixed)));

    % Display the registration result
    disp('Displaying registration result');
    figure, imshowpair(fixed, movingRegisteredAffineWithIC);
    title('Registration from affine model based on similarity initial condition.');

    % Overlay the registered DAPI and Hema images
    disp('Overlaying Hema and registered DAPI images');
    overlay = imfuse(fixed, movingRegisteredAffineWithIC, 'blend');
    figure, imshow(overlay);
    title('Overlay of Hema and registered DAPI');

    % Select region for cropping
    disp('Select a region for cropping');
    h = imrect;
    position = wait(h);
    top_left = round(position(1:2));
    bottom_right = round(position(1:2) + position(3:4));

    % Crop the images based on the selected region
    disp('Cropping images based on selected region');
    cropped_HE = imcrop(HE, [top_left, bottom_right - top_left]);
    cropped_Hema = imcrop(fixed, [top_left, bottom_right - top_left]);
    cropped_DAPI = imcrop(movingRegisteredAffineWithIC, [top_left, bottom_right - top_left]);
    cropped_BCL2 = imcrop(movingRegisteredAffineWithIC2, [top_left, bottom_right - top_left]);
    cropped_PAX5 = imcrop(movingRegisteredAffineWithIC3, [top_left, bottom_right - top_left]);

    % Save the cropped images
    disp('Saving cropped images');
    imwrite(cropped_HE, fullfile(he_output_path, strcat('registered_HE_', name, '.tif')));
    imwrite(cropped_DAPI, fullfile(dapi_output_path, strcat('registered_dapi_', name, '.tif')));
    imwrite(cropped_BCL2, fullfile(bcl2_output_path, strcat('registered_bcl2_', name, '.tif')));
    imwrite(cropped_PAX5, fullfile(pax5_output_path, strcat('registered_pax5_', name, '.tif')));

    disp(['Saved registered and cropped images for: ', name]);
end

disp('All files processed.');
