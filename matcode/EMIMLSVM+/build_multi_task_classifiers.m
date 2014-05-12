function [model, cv_parameters] = build_multi_task_classifiers(train_kernel_matrix, train_labels, task_rel)

% use Regularized Multi-Task Learning method to implement classifiers with considering relations
% between labels.
%
% function model = build_multi_classifiers(train_kernel_matrix, train_labels, ru)
%
% <notes>
%
% <mothods> 
% use primal method that all the alpha are calculated simutanously.
%
% ref. T. Evgeniou, M. Pontil, Regularized Multi-Task Learning. PAKDD'04.



[train_bag_num, label_num] = size(train_labels);
extend_label_num  = train_bag_num *label_num;

% check for validation of train_labels
elements = unique(train_labels);
if isempty(setdiff(elements, [0, 1])) == false
    error('labels must be 0 or 1.');
end


% reform the common part of the multi-task kernel
kernel_matrix = repmat(train_kernel_matrix, [label_num, label_num]); 


%% train a classifier 
cv_num = 10;                                                                    % use 10-fold cross-validation.
label_vector = train_labels(:);                                                 % reform the label matrix to a single label vector.

% set imbalance factor for each sample.
base_Cs = ones(label_num*train_bag_num, 1);
label_spc_idx = cell(label_num);
for i = 1:label_num
    label_spc_idx{i} = (i-1)*train_bag_num+1 : i*train_bag_num;                 % sample_idx;
    
    single_labels = train_labels(:, i);                                         % get specific labels.
    tmp_Cs = ones(train_bag_num, 1);

    tmp_Cs(single_labels==1) = sum(single_labels==0)/sum(single_labels==1);     % imbalance rate
    base_Cs(label_spc_idx{i}, 1) = tmp_Cs;
end

%<Note>, Cs should be adjusted to your real data.
Cs  = [1, 10, 50, 100, 150, 200];

if label_num == 1                                                               %if there is noly on label a task, then use ordianry svm.
    RUs = 0;                                                                    % when only single label, then U = Inf, RUs = 0;
else
    
    %<Note> Us should be adjusted to your real data. This is just for demo.
    Us  =  [task_rel^2, 0.1*task_rel^2];
    Us  =  label_num * Us  * 10;
    RUs =  1./Us;
end


best_cv = -1;
cv_process_values = [];                                                         %record the cv process for analysis.
cv_idx = 0;
for ui = 1:length(RUs)
    ru = RUs(ui);                                                               % get current 1/U.
    
    % reform the kernel used.
    tmp_kernel_matrix = [(1:extend_label_num)', ru * kernel_matrix];            % common part = (1/U) * kernel_matrix
    for i = 1:label_num
        tmp_kernel_matrix(label_spc_idx{i}, label_spc_idx{i}+1) = tmp_kernel_matrix(label_spc_idx{i}, label_spc_idx{i}+1) + train_kernel_matrix;
    end

    % perform a cross-validation for a specific (ru = 1/U)
    best_inner_cv = -1;                                                         % record the best cv for each ru = 1/U.
    for ci = 1:length(Cs)
        cc  = Cs(ci)/(1+ru);                                                            
        CCS = cc * base_Cs;

        cmd = ['-t 4 -v ', num2str(cv_num)];
        [cv_value, pt_labels] = svmtrain(full(label_vector), tmp_kernel_matrix, cmd, CCS);
        
        %record the cv process.
        cv_idx = cv_idx + 1;
        cv_process_values(cv_idx, 1) = Cs(ci);
        cv_process_values(cv_idx, 2) = ru;
        cv_process_values(cv_idx, 3) = cv_value;

        
        % record the CV result for a specific U
        if cv_value >= best_inner_cv
            best_inner_cv = cv_value;
            inner_C = cc;
        end
    end

    % select the overall best value for (1/U, C)
    if  best_inner_cv >= best_cv
        best_cv = best_inner_cv;                                                % the best overall cross validation
        C  = inner_C;                                                           % the best C
        RU = ru;                                                                % the best ru = 1/U.
    end
end


%record cv parameters.
cv_parameters.opt_ru  = RU;
cv_parameters.C       = C;
cv_parameters.cv_perf = best_cv;
cv_parameters.cv_process = cv_process_values;


%% train the final model.
fprintf('\nbuilding the final svm model with RU = %f, C = %f...\n', RU, C);
best_kernel_matrix = [(1:extend_label_num)', RU*kernel_matrix];
for i = 1:label_num
    best_kernel_matrix(label_spc_idx{i}, label_spc_idx{i}+1) = best_kernel_matrix(label_spc_idx{i}, label_spc_idx{i}+1) + train_kernel_matrix;
end

CCS = C * base_Cs;
bestcmd = '-t 4';
model   = svmtrain(full(label_vector), best_kernel_matrix, bestcmd, CCS);

