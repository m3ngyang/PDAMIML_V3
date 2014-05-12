function cvg = coverage(predictedY, ground_truth)
%==================================================================================================
% cvg = coverage(predictedY, ground_truth)
% 
%
% Li Yingxin, Aug. 26, 2008
%==================================================================================================
[sample_num, label_num] = size(ground_truth);

cvg = 0;
for i = 1:sample_num
    gt_yes = find(ground_truth(i,:) == 1);
    pt_yes = predictedY(i, gt_yes);
    
    [sorted, sorted_idx] = sort(predictedY(i,:), 'descend');

    ordered = [];
    for j = 1:length(pt_yes)
        ordered = [ordered, find(sorted == pt_yes(j))];
    end;

    cvg = cvg + max(ordered);
end

cvg = cvg/sample_num-1;