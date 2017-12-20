classdef DistillLoss < dagnn.Loss
  properties
    T1 = 0
    T2 = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nndistillloss(inputs{1}, inputs{2}, [], 'T1', obj.T1, 'T2', obj.T2) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + sum(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nndistillloss(inputs{1}, inputs{2}, derOutputs{1}, 'T1', obj.T1, 'T2', obj.T2) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function obj = DistillLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
