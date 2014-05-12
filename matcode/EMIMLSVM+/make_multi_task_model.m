function [predicted_labels, predicted_Y, task_models] = make_multi_task_model(task_info, ...
                                           train_kernel_matrix, test_kernel_matrix, ...
                                           train_labels, test_labels)

% the main entrance for multi-task learning.
% function [predicted_labels, predicted_Y] = make_multi_task_model(task_info, ...
%                                           train_kernel_matrix, test_kernel_matrix, ...
%                                           train_labels, test_labels)
%
% task_info: t by 2 cell. Rows of task_info correspond to a task.
%                         {label indices, ru=1/u for this task}
%


[test_bag_num, label_num] = size(test_labels);
%predicted_labels          = sparse(test_bag_num, label_num); 
predicted_labels          = zeros(test_bag_num, label_num); 
predicted_Y               = zeros(test_bag_num, label_num);
task_num                  = size(task_info, 1);                             %number of tasks
task_models               = cell(task_num, 2);                              %classifiers and parameters

for i = 1:task_num
    % the labels of a task, and its correlation value related weight.
    task_id    = i;
    label_ids  = task_info{task_id, 1};                                     %the label indices of a task
    task_rel   = task_info{i, 2};                                           
    
    %train a model for tasks of a cluster.
    fprintf('\nbuilding models for labels: ');
    fprintf('%d ', label_ids);
    fprintf('of task %d...\n', task_id);
    
    trn_labels        = train_labels(:, label_ids); 
    [model, cvp]      = build_multi_task_classifiers(train_kernel_matrix, trn_labels, task_rel);
    task_models{i, 1} = cvp;   %record cv parameter
    task_models{i, 2} = model; %record models
    
    %test for the model.
    task_test_labels = test_labels(:, label_ids);
    [pL, pY]   = classify_multi_task_miml(model, test_kernel_matrix, task_test_labels, cvp.opt_ru);
    
    %record task results
    predicted_labels(:, label_ids) = pL;
    predicted_Y(:, label_ids) = pY;
end



%% T-criterion

%for i = 1:size(predicted_labels, 1)
%    if sum(predicted_labels(i, :) == 1) == 0                                %all predicted labels are negative.
%        %find the position with maximum predicted value.
%        [max_value, max_idx] = max(predicted_Y(i, :));
%        
%        %set the label with maximum value.
%        predicted_labels(i, max_idx) = 1;
%    end
%end

end
