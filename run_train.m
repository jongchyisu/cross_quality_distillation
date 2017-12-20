function run_train(gpuidx, dataset_A, dataset_B, dirName, prefix_A, varargin)
%% Default learning parameters
% opts.learningRate = [0.001*ones(1,5), (0.001:-0.0001:0.0001), 0.0001*ones(1, 10), 0.00005*ones(1,10)]; % learning rate for CARS
opts.learningRate = [0.0005*ones(1,5), 0.0005:-0.00005:0.0001 0.0001*ones(1,10), 0.00005*ones(1,10)]; % learning rate for CUB and distilling verydeep
opts.batchSize = 128;
opts.numEpochs = 30;
opts.momentum = 0.9;
opts.weightDecay = 0.0005;
opts.T1 = 0;
opts.T2 = 0;
opts.lambda = 0;
opts.useVal = 0;
% derOutputs will be added in runtime at cnn_train_dag.m if using distillation!
opts.derOutputs = {'objective', 1};

%% Dataset parameters
opts.imdbDir = [];
opts.keepAspect = 0;
opts.useCurriculum = 0;
opts.useSRCNN = 0;
opts.useDistill = 0;
opts.useVgg = 1;
opts.model = 'vgg-m';
opts.useGpu = gpuidx;

% passing opts to here
[opts, varargin] = vl_argparse(opts,varargin) ;

% set pre-trained models
if ~opts.useDistill && ~opts.useVgg % only for trainAtuneB, here 'dataset_B' is actually the name of dataset A
    model_name = ['../', dirName, '/trainA/', dataset_B, '-seed-01/net-deployed.mat'];
else
    if strcmp(opts.model,'vgg-m')
        model_name = 'imagenet-vgg-m.mat';
    elseif strcmp(opts.model,'vgg-vd')
        model_name = 'imagenet-vgg-verydeep-16.mat';
    end
end

if opts.useSRCNN
    model_name = 'imagenet-vgg-m.mat';
end

% Get imdb and opts for network 1
[opts1, imdb] = model_setup('dataset', dataset_A, ...
              'prefix', fullfile(dirName, prefix_A), ...
              'model', model_name,...
              'batchSize', opts.batchSize, ...
              'learningRate', opts.learningRate, ...
              'momentum', opts.momentum, ...
              'weightDecay', opts.weightDecay, ...
              'numEpochs', opts.numEpochs, ...
              'useGpu', opts.useGpu, ...
              'rgbJitter', false, ...
              'useVal', opts.useVal, ...
              'useSRCNN', opts.useSRCNN, ...
              'networkType', 'dagnn', ...
              'keepAspect', opts.keepAspect, ...
              'useDistill', opts.useDistill, ...
              'imdbDir', opts.imdbDir, ...
              'T1', opts.T1, ...
              'T2', opts.T2, ...
              'lambda', opts.lambda);

% funtion of initialization of the first network
initNetFn1 = @initializeNetworkDag;

% train the first network
imdb_cnn_train(imdb, initNetFn1, @getBatchFn, opts1);

