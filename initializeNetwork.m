% -------------------------------------------------------------------------
function net = initializeNetwork(imdb, opts)
% -------------------------------------------------------------------------

if isfield(imdb.classes,'number') % for combining datasets
    numClass = sum(imdb.classes.number);
else
    numClass = length(imdb.classes.name);
end
    
%% add vgg-m
net = load(fullfile('data/models', opts.model)); % Load model if specified

% net.normalization.keepAspect = opts.keepAspect;
fprintf('Initializing from model: %s\n', opts.model);

if isfield(net,'params')
    net = dagnn.DagNN.loadobj(net);
    net.layers(net.getLayerIndex('fc7')).inputs = {'dropout1'}; %% need to be cell!
    dropout1_input_name = net.layers(net.getLayerIndex('relu6')).outputs;
    net.addLayer('dropout1', dagnn.DropOut(), dropout1_input_name, 'dropout1');
    net.layers(net.getLayerIndex('fc8')).inputs = {'dropout2'}; %% need to be cell!
    dropout2_input_name = net.layers(net.getLayerIndex('relu7')).outputs;
    net.addLayer('dropout2', dagnn.DropOut(), dropout2_input_name, 'dropout2');
    net.removeLayer('prob');
    net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'prediction','label'}, 'objective');
else
    % backup the fully connected layers
    tempLayers = net.layers(end-3:end);
    net.layers = net.layers(1:end-4);

    % Add dropout layer after fully connected layers
    net = add_dropout(net, '6');

    net.layers{end+1} = tempLayers{1};
    net.layers{end+1} = tempLayers{2};

    net = add_dropout(net, '7');

    % Initial the last but one layer with random weights
    wopts.scale = 1;
    wopts.weightInitMethod = 'gaussian';
    net.layers{end+1} = struct('type', 'conv', 'name', 'fc8', ...
                               'weights', {{init_weight(wopts, 1, 1, 4096, numClass, 'single'), zeros(numClass, 1, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0, ...
                               'learningRate', [10 20], ...
                               'weightDecay', [1 0]) ;

    % Last layer is softmaxloss (switch to softmax for prediction)
    net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

    % Rename classes
    net.meta.classes.name = imdb.classes.name;
    net.meta.classes.description = imdb.classes.name;

    % Other details
    net.meta.normalization.imageSize = [224, 224, 3] ;
    % net.meta.normalization.interpolation = 'bicubic' ;
    % net.meta.normalization.border = 256 - net.normalization.imageSize(1:2) ;
    % net.meta.normalization.averageImage = [] ;
    net.meta.normalization.keepAspect = opts.keepAspect ;
    net.meta.augmentation.rgbVariance = [];
    net.meta.augmentation.transformation = opts.dataAugmentation{1};
end

function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

function net = add_dropout(net, id)
% --------------------------------------------------------------------

net.layers{end+1} = struct('type', 'dropout', ...
                           'name', sprintf('dropout%s', id), ...
                           'rate', 0.5) ;

