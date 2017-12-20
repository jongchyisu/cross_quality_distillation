function imo = imdb_get_batch(images, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.useSRCNN = false ;% added for SRCNN
opts.useDistill = 0 ;% added for Distilling multi-dataset
opts.noise_ratio = 0;
opts.numEpochs = 0;
opts = vl_argparse(opts, varargin);

im = cell(numel(images),1);
for i=1:numel(images)
    im{i} = single(imread(images{i})) ;
end

if (~isempty(opts.rgbVariance) && isempty(opts.averageImage))
    opts.averageImage = zeros(1,1,3) ;
end

if numel(opts.averageImage) == 3
    opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
            numel(images)*opts.numAugments, 'single') ;
        
for i=1:numel(images)
    
    % acquire image
    imt = im{i} ;
    
    if size(imt,3) == 1
        imt = cat(3, imt, imt, imt) ;
    end
    
    % resize
    w_ori = size(imt,2) ;
    h_ori = size(imt,1) ;
    factor = [(opts.imageSize(1)+opts.border(1))/h_ori ...
              (opts.imageSize(2)+opts.border(2))/w_ori];
    if opts.keepAspect
        factor = max(factor) ;
        imt = imresize(imt, 'scale', factor) ; % use bicubic default
    else
        imt = imresize(imt, opts.imageSize(1:2)) ; % use bicubic default
    end
    
    % crop & flip
    if (~isempty(opts.averageImage) && ~opts.useSRCNN)
        offset = opts.averageImage ;
        if ~isempty(opts.rgbVariance)
            offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
        end
        imo(:,:,:,i) = bsxfun(@minus, imt, offset) ;
    else
        imo(:,:,:,i) = imt ;
    end
end
