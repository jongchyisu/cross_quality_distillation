function [opts, imdb] = model_setup(varargin)

% setup ;

opts.derOutputs = {'objective', 1};%, 'distilled_loss', 0.01};
opts.networkType = 'simplenn';
opts.numFetchThreads = 10;
opts.seed = 1 ;
opts.batchSize = 128 ;
opts.numEpochs = 100;
opts.momentum = 0.9;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005;
opts.keepAspect = true;
opts.useVal = false;
opts.useGpu = 1 ;
opts.regionBorder = 0.05 ;
opts.printDatasetInfo = false ;
opts.excludeDifficult = true ;
opts.datasetSize = inf;
opts.dataset = 'cub' ;
opts.carsDir = 'data/cars';
opts.cubDir = 'data/cub';
opts.suffix = 'baseline' ;
opts.prefix = 'v1' ;
opts.model  = 'imagenet-vgg-m.mat';
opts.rgbJitter = false;
opts.dataAugmentation = {'none', 'none', 'none'};
opts.cudnn = true;
opts.batchNormalization = false;
opts.cudnnWorkspaceLimit = 1024*1024*1024; 
opts.T1 = 1;
opts.T2 = 1;
opts.lambda = 1;
opts.useSRCNN = 0;
opts.useDistill = 0;

[opts, varargin] = vl_argparse(opts,varargin) ;

opts.expDir = sprintf('data/%s/%s-seed-%02d', opts.prefix, opts.dataset, opts.seed) ;
opts.imdbDir = fullfile(opts.expDir, 'imdb') ;
opts.resultPath = fullfile(opts.expDir, sprintf('result-%s.mat', opts.suffix)) ;

opts = vl_argparse(opts,varargin) ;

if isempty(opts.imdbDir)
    opts.imdbDir = fullfile(opts.expDir, 'imdb') ;
end

if nargout <= 1, return ; end

% % Setup GPU if needed
% if opts.useGpu
%   gpuDevice(opts.useGpu) ;
% end



% -------------------------------------------------------------------------
%                                                              Load dataset
% -------------------------------------------------------------------------

vl_xmkdir(opts.expDir) ;
vl_xmkdir(opts.imdbDir) ;

imdbPath = fullfile(opts.imdbDir, sprintf('imdb-seed-%d.mat', opts.seed)) ;
optsPath = fullfile(opts.expDir, sprintf('opts.mat')) ;
save(optsPath, '-struct', 'opts') ;
if exist(imdbPath)
  imdb = load(imdbPath) ;
%   if(opts.rgbJitter)
%       opts.pca = imdb_compute_pca(imdb, opts.expDir);
%   end
  return ;
end

switch opts.dataset
    case 'cub'
        imdb = cub_get_database(opts.cubDir, opts.useVal, opts.dataset);
    case 'cubcrop'
        imdb = cub_get_database(opts.cubDir, opts.useVal, opts.dataset);
    case 'cublowres'
        imdb = cub_get_database(opts.cubDir, opts.useVal, opts.dataset);
    case 'cubgray'
        imdb = cub_get_database(opts.cubDir, opts.useVal, opts.dataset);
    case 'cubedges'
        imdb = cub_get_database(opts.cubDir, opts.useVal, opts.dataset);
    case 'cubdistort'
        imdb = cub_get_database(opts.cubDir, opts.useVal, opts.dataset);
    case 'cubSR'
        imdb = cub_get_database(opts.cubDir, opts.useVal, opts.dataset);
    case 'cars'
        imdb = cars_get_database(opts.carsDir, opts.useVal, opts.dataset);
    case 'carscrop' % not using now
        imdb = cars_get_database(opts.carsDir, opts.useVal, opts.dataset);
    case 'carslowres'
        imdb = cars_get_database(opts.carsDir, opts.useVal, opts.dataset);
    case 'carsgray'
        imdb = cars_get_database(opts.carsDir, opts.useVal, opts.dataset);
    case 'carsedges'
        imdb = cars_get_database(opts.carsDir, opts.useVal, opts.dataset);
    case 'carsSR'
        imdb = cars_get_database(opts.carsDir, opts.useVal, opts.dataset);
    case 'aircraft-variant'
        imdb = aircraft_get_database(opts.aircraftDir, 'variant');
    case 'imagenet'
        imdb = cnn_imagenet_setup_data('dataDir', opts.ilsvrcDir);
    case 'imagenet-224'
        imdb = cnn_imagenet_setup_data('dataDir', opts.ilsvrcDir_224);
    otherwise
        error('Unknown dataset %s', opts.dataset) ;
end

save(imdbPath, '-struct', 'imdb') ;

% if(opts.rgbJitter)
%    opts.pca = imdb_compute_pca(imdb, opts.expDir);
% end

if opts.printDatasetInfo
  print_dataset_info(imdb) ;
end

% -------------------------------------------------------------------------
function [model, modelPath] = get_cnn_model_from_encoder_opts(encoder)
% -------------------------------------------------------------------------
p = find(strcmp('model', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = {[m e]} ;
  modelPath = encoder.opts{p+1};
else
  model = {} ;
  modelPath = {};
end

% bilinear cnn models
p = find(strcmp('modela', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
  modelPath = horzcat(modelPath, encoder.opts{p+1});
end
p = find(strcmp('modelb', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
  modelPath = horzcat(modelPath, encoder.opts{p+1});
end


