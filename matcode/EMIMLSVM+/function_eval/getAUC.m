function [auc, convex_auc] = getAUC(predictedY, ground_truth)

% calculate the area under ROC curve. for multi-label, return the means of all the single-label.
% [auc, convex_auc] = getAUC(predictedY, ground_truth)
% 
% predictedY  : n by k matrix. each row corresponds to a sample, and each column corresponds to a specific label.
% ground_truth: n by k matrix.
%
% auc         : the AUC with step hull.
% convex_auc  : the AUC with convex hull.


label_num = size(predictedY, 2);
gt_num    = size(ground_truth, 2);

if (label_num ~= gt_num)
    error('mismatch between predicted and ground truth.');
end

aucs = zeros(2, label_num);
for j = 1:label_num
    Y = predictedY(:, j);
    T = ground_truth(:, j);
    
    [tp, fp ] = roc(T, Y);                                % for calculating AUC without convex hull.                                          
    [tp1,fp1] = rocch(T,Y);                               % for calculating AUC with convex hull.          
    
    aucs(1, j)  = auroc(tp,fp);                           % AUC without convex hull, stardard, which has lower value
    aucs(2, j)  = auroc(tp1, fp1);                        % AUC with convex hull. which has larger value than auroc(tp, fp)
end


% calculating the mean of step auc and convex auc.
auc = mean(aucs(1, :));
convex_auc = mean(aucs(2, :));