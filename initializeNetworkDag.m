function net = initializeNetworkDag(imdb, opts)

net = dagnn.DagNN();
netb = initializeNetwork(imdb, opts);

if opts.useSRCNN
    % Load SRCNN model
    model = 'x4.mat';
    load(model);
    [conv1_patchsize2,conv1_filters] = size(weights_conv1);
    conv1_patchsize = sqrt(conv1_patchsize2);
    [conv2_channels,conv2_patchsize2,conv2_filters] = size(weights_conv2);
    conv2_patchsize = sqrt(conv2_patchsize2);
    [conv3_channels,conv3_patchsize2] = size(weights_conv3);
    conv3_patchsize = sqrt(conv3_patchsize2);
    weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, 1, conv1_filters); % 9 9 1 64
    weights_conv2 = permute(weights_conv2, [2 1 3]);
    weights_conv2 = reshape(weights_conv2, conv2_patchsize, conv2_patchsize, conv2_channels, conv2_filters); % 5 5 64 32
    weights_conv3 = permute(weights_conv3, [2 1]);
    weights_conv3 = reshape(weights_conv3, conv3_patchsize, conv3_patchsize, conv3_channels, 1); % 5 5 32 1
 
    % Add SRCNN layers
    net.addLayer('rgb2lcbcr', RGB2YCbCr, 'input', 'YCbCr') ;
    net.addLayer('seperate1', SeperateChannels, 'YCbCr', {'Y','CbCr'}) ;

    learningRate = [50,50];    
    net = addConvLayer(net, 'conv1', 'Y', 'conv1', single(weights_conv1), single(biases_conv1), 1, 1, 0, [1,1], learningRate);
    net.addLayer('relu1', dagnn.ReLU(), 'conv1', 'relu1'); 
    net = addConvLayer(net, 'conv2', 'relu1', 'conv2', single(weights_conv2), single(biases_conv2), 1, 1, 0, [1,1], learningRate);
    net.addLayer('relu2', dagnn.ReLU(), 'conv2', 'relu2'); 
    net = addConvLayer(net, 'conv3', 'relu2', 'conv3', single(weights_conv3), single(biases_conv3), 1, 1, 0, [1,1], learningRate);
    
    net.addLayer('Y_padding', PadSRCNN, {'conv3', 'Y'}, 'Y_up') ;
    
    net.addLayer('concat1', dagnn.Concat, {'Y_up', 'CbCr'}, 'YCbCr_up') ;
    net.addLayer('lcbcr2rgb', YCbCr2RGB, 'YCbCr_up', 'RGB_up') ;
    
    % Load averageImage from vgg-m
    averageImage = single(netb.meta.normalization.averageImage);
    net.addLayer('subtractMean', SubtractMean('averageImage', averageImage), 'RGB_up', 'netb_input') ;
end
    
% Add vgg-m on top
if ~isa(netb, 'dagnn.DagNN')
    netb = dagnn.DagNN.fromSimpleNN(netb, 'canonicalNames', true) ;
end

% If use SRCNN, transfer vgg-m from netb to net. Otherwise just load vgg-m.
if opts.useSRCNN
    for i=1:numel(netb.layers)
        layerName = strcat('netb_', netb.layers(i).name);
        if i < numel(netb.layers)
            input =  strcat('netb_', netb.layers(i).inputs);
        else
            input =  netb.layers(i).inputs;
        end
        if i < numel(netb.layers)-1
            output = strcat('netb_', netb.layers(i).outputs);
        elsehasBias
            output = netb.layers(i).outputs;
        end
        params = strcat('netb_', netb.layers(i).params);
        net.addLayer(layerName, netb.layers(i).block, input, output, params);

        for f = 1:numel(params)
            varId = net.getParamIndex(params{f});
            varIdb = netb.getParamIndex(netb.layers(i).params{f});
            if strcmp(net.device, 'gpu')
                net.params(varId).value = gpuArray(netb.params(varIdb).value);
            else
                net.params(varId).value = netb.params(varIdb).value;
            end
            net.params(varId).learningRate = netb.params(varIdb).learningRate;
            net.params(varId).weightDecay = netb.params(varIdb).weightDecay;
        end
    end
    % add meta from vgg-m
    net.meta = netb.meta;
else
    net = netb;
end

% Add prediction and loss layers
if opts.useDistill
    distillBlock = DistillLoss('T1', opts.T1,'T2', opts.T2);
    net.addLayer('distill', distillBlock, ...
                 {'prediction','target_prob'}, 'distilled_loss') ;
end
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1err') ;
net.rebuild();


function net = addConvLayer(net, layerName, input, output, init_weights_f, init_weights_b, hasBias, stride, pad, weightDecay, learningRate)
% --------------------------------------------------------------------
block = dagnn.Conv('hasBias', hasBias, 'stride', stride, 'pad', pad);    
net.addLayer(layerName, block, input, output, {[layerName,'_f'], [layerName,'_b']}) ;
idx = net.getParamIndex([layerName,'_f']);
net.params(idx).weightDecay = weightDecay(1); 
net.params(idx).learningRate = learningRate(1);
net.params(idx).value = single(init_weights_f);
idx = net.getParamIndex([layerName,'_b']);
net.params(idx).weightDecay = weightDecay(2); 
net.params(idx).learningRate = learningRate(2);
net.params(idx).value = single(init_weights_b);

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
