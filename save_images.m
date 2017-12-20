function save_images(data_dir, useVal, dataset, degrade)

% Select dataset
data_dir = 'data';
useVal = false;
dataset = 'cub' % choose from 'cub','cars'
degrade = 'distort' % choose from 'edges','gray','lowres','distort', and 'lowresfull' only for cub
dataset_dir = fullfile(data_dir,dataset)

% Create other degraded images
% Download and put CUB dataset in data/cub/images for original images
% and data/cub/images_cropped_keepAp for cropped images
% Download and put CARS dataset in data/cars/car_ims
if strcmp(dataset,'cub')
    if strcmp(dataset,'lowresfull')
        ori_dataset = 'cub'
    else
        ori_dataset = 'cubcrop'
    end
    imdb_ori = cub_get_database(dataset_dir, useVal, ori_dataset);
    imdb_degraded = cub_get_database(dataset_dir, useVal, strcat(dataset,degrade));
else
    imdb_ori = cars_get_database(dataset_dir, useVal, 'cars');
    imdb_degraded = cars_get_database(dataset_dir, useVal, strcat(dataset,degrade));
end

inDir = imdb_ori.imageDir;
outDir = imdb_degraded.imageDir;
if ~exist(outDir, 'dir')
    mkdir(outDir)
end

if strcmp(degrade,'edges')
    % Setup Structured Edge Detector (Dollar and Zitnick)
    % Download and setup their code first
    % Code is borrowed from their demo code
    addpath(genpath('../toolbox'))
    addpath(genpath('../edges'))
    opts2=edgesTrain();                % default options (good settings)
    opts2.modelDir='models/';          % model will be in models/forest
    opts2.modelFnm='modelBsds';        % model name
    opts2.nPos=5e5; opts2.nNeg=5e5;     % decrease to speedup training
    opts2.useParfor=1;                 % parallelize if sufficient memory

    tic, model=edgesTrain(opts2); toc; % will load model if already trained
    cd('../distill-net')

    model.opts.multiscale=0;          % for top accuracy set multiscale=1
    model.opts.sharpen=2;             % for top speed set sharpen=0
    model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
    model.opts.nThreads=4;            % max number threads for evaluation
    model.opts.nms=0;                 % set to true to enable nms
elseif strcmp(degrade,'distort')
    % Download and setup tpsWarp first
    addpath('../tpsWarp');
end

if strcmp(dataset,'cub')
    num_images = numel(imdb_ori.images.name);
else
    image_names = dir([inDir,'/*.jpg']);
    num_images = image_names;
end

%% Generate low quality images
for i=1:num_images
    disp(i);

    if strcmp(dataset,'cub')
        im = imread(fullfile(inDir, imdb_ori.images.name{i}));
        [fpath, fname, ~] = fileparts(imdb_ori.images.name{i});
        fname = [fname, '.jpg'];
    else
        name = image_names(i).name;
        im = imread(fullfile(inDir, name));
    end

    if strcmp(dataset,'cub')
        if ~exist(fullfile(outDir, fpath), 'dir')
            mkdir(fullfile(outDir, fpath))
        end
    end

    switch degrade
    case {'lowres','lowresfull'}
        ori_size = size(im);
        im = imresize(im,[50,50]);
    case 'gray'
        if size(im,3)>1
            im = rgb2gray(im);
        end
    case 'edges'
        if size(im,3)>1
            im = edgesDetect(im,model);
        else % CUB dataset has 3 gray images!
            im = edgesDetect(cat(3,im,im,im),model);
        end
        im = im*255;
    case 'distort'
        [h,w,d] = size(im);
        % resize
        factor = [224/h, 224/w];
        factor = max(factor) ;
        im = imresize(im, 'scale', factor, 'method', 'bilinear') ;
        y_c = 8:16:224;
        x_c = 8:16:224;
        [X,Y] = meshgrid(x_c,y_c);
        center = [Y(:),X(:)];
        X_n=X+4*randn(size(X));
        Y_n=Y+4*randn(size(Y));
        target = [Y_n(:), X_n(:)];
        interp = [];
        interp.method = 'nearest';
        interp.radius=8;
        interp.power=1;
        [im, ~, ~] = tpswarp(im, [224 224], center, target, interp);
    case 'SR'
        ori_size = size(im);
        im = imresize(im,[56,56]);
        up_scale = 4;
        model = '../SRCNN/model/9-5-5(ImageNet)/x4.mat';
        % work on illuminance only
        if size(im,3)>1
            im_ycbcr = rgb2ycbcr(im);
            im_y = im_ycbcr(:, :, 1);
        else % CUB has 3 gray images!
            im_y = im;
        end
        % im_gnd = modcrop(im_y, up_scale);
        im_gnd = single(im_y)/255;
        % bicubic interpolation
        im_l = im_gnd;%imresize(im_gnd, 1/up_scale, 'bicubic');
        im_b = imresize(im_l, up_scale, 'bicubic');
        % SRCNN
        im_h = SRCNN(model, im_b);
        % remove border
        im_h = uint8(im_h * 255);
        % resize to original size
        im_h = imresize(im_h,[224, 224], 'bicubic');
        if size(im,3)>1
            im_ycbcr = imresize(im_ycbcr,[224, 224], 'bicubic');
            im_h = cat(3, im_h, im_ycbcr(:, :, 2:3));
            im_h = ycbcr2rgb(im_h);
        end
        im = im_h;
    end

    % resized to 224 so no need to do resizing online
    im = imresize(im,[224,224]);
    if strcmp(dataset,'cub')
        imwrite(uint8(im),fullfile(outDir, fpath, fname),'jpg');
    else
        imwrite(uint8(im),fullfile(outDir, name),'jpg');
    end
end
