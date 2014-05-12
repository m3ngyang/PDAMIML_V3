function [predicted_labels, predicted_Y] = classify_multi_task_miml(classifier, kernel_matrix, test_labels, ru)

% make predictions with multi-task classifier
% function [pt_labels, pt_y] = classify_multi_task(classifier, kernel_matrix, test_labels, ru)
% <parameters>
% kernel_matrix : the kernel between test bags(m) and training bags(n). e.g. K(m*n).


%% reform the common part of a new kernel.
[test_bag_num, label_num] = size(test_labels);
train_bag_num = size(kernel_matrix, 2);

kernel_matrix_st = zeros(test_bag_num, train_bag_num*label_num + 1);
kernel_matrix_st(:, 1)     = (1:test_bag_num)';
kernel_matrix_st(:, 2:end) = ru * repmat(kernel_matrix, 1, label_num);


%% calculate the outputs
predicted_labels = sparse(test_bag_num, label_num);       %use sparse matrix to store predicted labels.
predicted_Y      = zeros(test_bag_num,  label_num);

for i = 1:label_num
    % reform a new test-train kernel.
    test_train_kernel = kernel_matrix_st;
    label_spc_idx     = (i-1)*train_bag_num+1 : i*train_bag_num;
    test_train_kernel(:, label_spc_idx+1) = test_train_kernel(:, label_spc_idx+1) + kernel_matrix;
    
    % make prediction by svm.
    [pt_label, accuracy, ptY] = svmpredict(test_labels(:, i), test_train_kernel, classifier);
    
    % correct the output value. 
    %first_label_output = pt_label(1);
    %if first_label_output == 0
    %    first_label_output = -1;
    %end
    
    %if first_label_output*ptY(1) < 0
    %   ptY = -ptY;
    %end
     
    % record results.
    predicted_Y(:, i)      = ptY;
    predicted_labels(:, i) = pt_label;
end
