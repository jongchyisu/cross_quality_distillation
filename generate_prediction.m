function generate_prediction(imdb, opts, net, outputDir)
%-------------------------------------------------------------------------------------
% compute the output of first network
%-------------------------------------------------------------------------------------    
% getBatch function
getBatch = getBatchFn(opts, net.meta);

% directory for saving outputs
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end

isDag = 0;
% check if it's a dagNN
if isfield(net, 'params')
    net = dagnn.DagNN.loadobj(net);
    isDag = 1;
    if opts.useGpu
        net.move('gpu');
    end
end

% remove softmax layer because we want logits
net.removeLayer('prob');

% get the outputs
for i=1:numel(imdb.images.name) % fixed, using parfor now

    [fpath, fname, ~] = fileparts(imdb.images.name{i});
    fname = [fname, '.mat'];

    if ~exist(fullfile(outputDir, fpath), 'dir')
        mkdir(fullfile(outputDir, fpath))
    end

    fprintf('extracting feature of %d/%d from net 1\r', i, numel(imdb.images.name))
    if exist(fullfile(outputDir, fpath, fname), 'file')
        continue;
    end

    if ~isDag % use simplenn
%             [im, ~] = getBatch(imdb, i);
        [im, ~] = feval(getBatch, imdb, i);
        res = vl_simplenn(net, im, [], [], ...
            'accumulate', false, ...
            'conserveMemory', false, ...
            'cudnn', true) ;
        prob = res(end-1).x;
    else % use dagnn
        inputs = getBatch(imdb, i);
        inputs = inputs(1:2);
        net.eval(inputs)
        prob = net.vars(net.getVarIndex('prediction')).value;
    end
    if opts.useGpu
        prob = gather(prob);
    end
    save_parfor(outputDir, fpath, fname, prob);
end
end

function save_parfor(outputDir, fpath, fname, prob)
    save(fullfile(outputDir, fpath, fname), 'prob');
end