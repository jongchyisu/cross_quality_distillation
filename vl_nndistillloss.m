function Y = vl_nndistillloss(X,v,dzdy,varargin)
% X is prediction, having the same size as v

opts.T1 = 1;
opts.T2 = 1;
opts = vl_argparse(opts, varargin);

if isa(X,'gpuArray')
  dataType = classUnderlying(X) ;
else
  dataType = class(X) ;
end
switch dataType
  case 'double', toClass = @(x) double(x) ;
  case 'single', toClass = @(x) single(x) ;
end

% predected output and target output size should be the same
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
sz_ = [size(v,1) size(v,2) size(v,3) size(v,4)] ;
assert(isequal(sz_, sz)) ;

% compute distillation loss
prob_v = getprob(v, opts.T1);
prob = getprob(X, opts.T2);
if isempty(dzdy)
    Y = -sum(prob_v.*log(prob), 3);
    % use L-2 for logits
%     Y = sum((X-v).*(X-v), 3)/(2*mean(sum(v.*v, 3)));
else
    delta = prob/ opts.T2 - prob_v/ opts.T1;
    Y = bsxfun(@times, delta, dzdy);
     % use L-2 for logits
%     Y = bsxfun(@times, X-v, dzdy)/(mean(sum(v.*v, 3)));
end
end

function Y = getprob(X, T)
    Xmax = max(X/T,[],3);
    ex = exp(bsxfun(@minus, X/T, Xmax));
    Y = bsxfun(@rdivide, ex, sum(ex,3));
end