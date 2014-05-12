function hamming_loss = getHamming_loss(predict_label, ground_truth)
%==========================================================================
% calculating hamming loss (percentage of mismatched numbers).
% hamming_loss = getHamming_loss(predict_label, ground_truth)
%
% Li Yingxin, Aug. 18, 2008.
%==========================================================================

mismatched    = sum(sum(predict_label ~= ground_truth));
[row, col]    = size(ground_truth);
hamming_loss  = mismatched/(row*col);  
