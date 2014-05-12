function errors = one_error_t(predictedY, ground_truth)
%==========================================================================
% get one_error: errors = one_error_t(predicted, ground_truth)
% on T-criterion sumption that a sample must have one positive label.
% if all predicted of a sample are negtive and the max predicted are
% positive.
% Li Yingxin, Aug. 26, 2008
%==========================================================================

[sample_num, label_num] = size(ground_truth);
errors = 0;

for i = 1:sample_num
    [max_value, max_idx] = max(predictedY(i, :));
    
    if ground_truth(i, max_idx) ~= 1
        errors = errors + 1;
    end
end

errors = errors/sample_num;