
% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.useGpu) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;

bopts.keepAspect = opts.keepAspect ;
bopts.useSRCNN = opts.useSRCNN;
bopts.useDistill = opts.useDistill;
if isfield(opts,'numEpochs')
    bopts.numEpochs = opts.numEpochs;
end

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(imdb,batch,varargin) getSimpleNNBatch(bopts,imdb,batch,varargin) ;
  case 'dagnn'
    fn = @(imdb,batch,varargin) getDagNNBatch(bopts,useGpu,imdb,batch,varargin) ;
end


% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch, varargin)
% -------------------------------------------------------------------------
%% not using anymore
if isfield(imdb.classes,'number') % for combining datasets
    images = cell(1,size(batch,2));
    for i = 1:size(batch,2)
        images(i) = strcat([imdb.imageDir{imdb.images.ds(batch(i))} filesep], imdb.images.name(batch(i))) ;
    end
else
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
end

isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = imdb_get_batch(images, opts, ...
                      'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = imdb_get_batch(images, opts, ...
                      'prefetch', nargout == 0, ...
                      'transformation', 'none') ;
end

if nargout > 0
  labels = imdb.images.label(batch);
end


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch, varargin)
% -------------------------------------------------------------------------
if iscell(imdb.imageDir) % for combining datasets A+B
    images = cell(1,size(batch,2));
    for i = 1:size(batch,2)
        images(i) = strcat([imdb.imageDir{imdb.images.ds(batch(i))} filesep], imdb.images.name(batch(i))) ;
    end
else
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
end

isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
    im = imdb_get_batch(images, opts, ...
                        'prefetch', nargout == 0) ;
else
    % validation: disable data augmentation
    im = imdb_get_batch(images, opts, ...
                        'prefetch', nargout == 0, ...
                        'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end

inputs = {'input', im, 'label', imdb.images.label(batch)} ;
if opts.useDistill % if has distillation loss
    target = zeros(1, 1, numel(imdb.classes.name), numel(batch));
    for i=1:numel(batch)
        imgName = imdb.images.name{batch(i)};
        matName = [imgName(1:end-4) '.mat'];
        load(fullfile(imdb.netOutputDir{1}, matName));
        target(:,:,:,i) = prob;
    end
    inputs = cat(2, inputs, {'target_prob', target});
end

end
