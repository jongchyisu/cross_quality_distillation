function [acc, confusion]= run_test(imdb, net, gpuidx, opts)
% get testing accuracy and confusion matrix for trained model 

useGpu = gpuidx;

% setting for getBatchFn
gbopts.useGpu = opts.useGpu;
gbopts.numFetchThreads = 1;
gbopts.useSRCNN = opts.useSRCNN;
gbopts.keepAspect = opts.keepAspect;
gbopts.useDistill = opts.useDistill;

isDag = isfield(net, 'params') || isobject(net);

test = find(imdb.images.set == 2);
gt = imdb.images.label(test);

if isDag
    net = dagnn.DagNN.loadobj(net);

    net.mode = 'test';

    gbopts.networkType = 'dagnn';
    
    if useGpu
        net.move('gpu')
    end
    
    getBatch = getBatchFn(gbopts, net.meta);
    
    probName = 'prob';
    
    pred_prob = cell(numel(test),1);
    for i=1:numel(test)
        fprintf('testing image: %d/%d\r', i, numel(test));
        inputs = getBatch(imdb, test(i), 1);
        inputs = inputs(1:2);
        net.eval(inputs)
        
        prob = net.vars(net.getVarIndex(probName)).value;
        if useGpu
            pred_prob{i} = squeeze(gather(prob));
        else
            pred_prob{i} = squeeze(prob);
        end
    end
    
    net.move('cpu')
else
    %% Note: not using simplenn anymore, not good for the latest version.
    gbopts.networkType = 'simplenn';
    
    if useGpu
        net = vl_simplenn_move(net, 'gpu') ;
    end    
    
    getBatch = getBatchFn(gbopts, net.meta);
    
    pred_prob = cell(numel(test),1);
    for i=1:numel(test)
        fprintf('testing image: %d/%d\r', i, numel(test));
        [im, ~] = getBatch(imdb, test(i));
        
        if useGpu
            im = gpuArray(im);
        end
        res = vl_simplenn(net, im, [], [], ...
            'mode', 'test', ...
            'cudnn', true) ;
        
        prob = res(end).x;
        
        if useGpu
            pred_prob{i} = squeeze(gather(prob));
        else
            pred_prob{i} = squeeze(prob);
        end
    end
end

% Compute pred_probictions, confusion and accuracy
pred_prob = cat(2, pred_prob{:});
[~,pred] = max(pred_prob(min(gt):max(gt),:),[],1) ;

[confusion, acc] = compute_confusion(size(pred_prob, 1), gt, pred, 1, true) ; % true for per image accuracy, false for per class
fprintf('Accuracy: %.2f%%\n', acc*100)
% imagesc(confusion);colormap gray;
