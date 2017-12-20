function net = imdb_cnn_train(imdb, initNetFn, getBatchPointer, opts, varargin)
% Train a CNN model on a dataset supplied by imdb

opts.lite = false ;
% opts.numFetchThreads = 10 ;
opts.train.batchSize = opts.batchSize ;
opts.train.numEpochs = opts.numEpochs ;
opts.train.continue = true ;
opts.train.gpus = opts.useGpu ;
opts.train.prefetch = false ;
opts.train.learningRate = opts.learningRate ;
opts.train.weightDecay = opts.weightDecay ;
opts.train.momentum = opts.momentum ;
opts.train.expDir = opts.expDir ;
opts.train.derOutputs = opts.derOutputs;
opts.train.T1 = opts.T1;
opts.train.T2 = opts.T2;
opts.train.lambda = opts.lambda;
opts.train.useDistill = opts.useDistill;
% opts.networkType = 'simplenn';
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = initNetFn(imdb, opts) ;

if opts.rgbJitter
    % Compute image statistics (mean, RGB covariances, etc.)
    imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
    if exist(imageStatsPath)
        load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    else
        [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
        save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    end
    
    % Set the image average (use either an image or a color)
    %net.meta.normalization.averageImage = averageImage ;
    %net.meta.normalization.averageImage = rgbMean ;
    
    % Set data augmentation statistics
    [v,d] = eig(rgbCovariance) ;
    net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
    clear v d ;
else
    net.meta.augmentation.rgbVariance = [];
end

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------


switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchPointer(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;
% [net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true) ;

% Save model
%net = vl_simplenn_move(net, 'cpu');
%saveNetwork(fullfile(opts.expDir, 'final-model.mat'), net);

net = deploy_network(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
opts.networkType = 'simplenn' ;
fn = getBatchFn(opts, meta) ;
avg = {}; rgbm1 = {}; rgbm2 = {};

for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{end+1} = mean(temp, 4) ;
  rgbm1{end+1} = sum(z,2)/n ;
  rgbm2{end+1} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
