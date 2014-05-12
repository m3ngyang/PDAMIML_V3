function [macro_F1, micro_F1, contingency_matrix] = getF1(predict_labels, ground_truth)

% calculate F1 values for classification results.
% *[macro_F1, micro_F1, contingency_matrixs] = getF1(predict_labels, ground_truth)
%
% <inputs>
% predict      : n by k matrix. each row is an observation, each col is a label.
% ground_truth : n by k matrix. 
%
% Ying-Xin Li, June, 7. 2009.

label_num = size(ground_truth, 2);
contingency_matrix = cell(6, label_num+1);

contingency_matrix{1, 1} = 'p->p';
contingency_matrix{2, 1} = 'n->p';
contingency_matrix{3, 1} = 'p->n';
contingency_matrix{4, 1} = 'n->n';
contingency_matrix{5, 1} = 'F1';
contingency_matrix{6, 1} = 'single prediction';
contingency_matrix{7, 1} = 'p-rate';

for j = 1:label_num
    predict_each_class = predict_labels(:, j);
    ground_truth_class = ground_truth(:, j);
    
    pred1 = find(predict_each_class == 1);                          %predicted to be positive.
    pred0 = find(predict_each_class ~= 1);                          %predicted to be negative.
    
    gt1   = find(ground_truth_class == 1);                          %ground truth is positive.
    gt0   = find(ground_truth_class ~= 1);                          %ground truth is negative.  
   
    A = length(intersect(pred1, gt1));                              %precision. positive samples predicted as positive
    B = length(intersect(pred1, gt0));                              %false positive. negative samples predicted as positive.  
    
    C = length(intersect(pred0, gt1));                              %false negative. positive samples predicted as negative.   
    D = length(intersect(pred0, gt0));                              %specificity. negative samples predicted as negative.
    
    F1 = 0;
    if A ~= 0 || B ~= 0 || C ~= 0
        F1 = (2*A)/(2*A + B + C);
    end
    
    %if single_prediction == 1 means that the predicted labels are all set to 1 or -1. 
    single_prediction = 0;
    if (A+B ==0) || (C+D==0)
        single_prediction = 1;
    end
    
    %record the results for j-th label
    contingency_matrix{1, j+1} = A;
    contingency_matrix{2, j+1} = B;
    contingency_matrix{3, j+1} = C;
    contingency_matrix{4, j+1} = D;
    contingency_matrix{5, j+1} = F1;
    contingency_matrix{6, j+1} = single_prediction;
    contingency_matrix{7, j+1} = length(gt1)/(length(gt1)+length(gt0));
end

%macro_F1: the averaged performance in terms of F1. 
macro_F1 = mean(cell2mat(contingency_matrix(5, 2:end)));


%calculate micro_F1: the overall performance in terms of F1
A_all = sum(cell2mat(contingency_matrix(1, 2:end)));
B_all = sum(cell2mat(contingency_matrix(2, 2:end)));
C_all = sum(cell2mat(contingency_matrix(3, 2:end)));

micro_F1 = 0;
if A_all ~= 0 || B_all ~= 0 || C_all ~= 0
    micro_F1 = (2*A_all)/(2*A_all + B_all + C_all);
end
