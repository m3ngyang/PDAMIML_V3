function class_results = Main_MIMLSVM()

dirname = './results';
if exist(dirname, 'dir') == false
    mkdir(dirname);
end

result_file_name = [dirname, '/prediction_results.mat'];

               
%% dir settings
orginal_path = path();      
path('../libsvm_ci/', path);                                              %used libsvm_liyx instead of the original libsvm
path('./function_eval/', path);                                             %performance evaluation functions.


%% load kernels
KL_pair = ['./KernelLabelPairs/GO_0008094_lg_5.mat'];
load(KL_pair);
K = miKernel;
curLabels = labels;

%% load label informations
%index_name = ['../demo_data/idx_', num2str(demo_stage_range), '_', num2str(demo_term_nums), '.mat'];
[trIdx, teIdx] = gen_ind(size(curLabels,1));

[number_of_bags, label_num] = size(curLabels); %number of bags
exp_num   = size(trIdx, 2);                    %number of splits


%% main process
%class_results = cell(exp_num+2, 10);
class_results = cell(exp_num+2, 7);

class_results{1, 1} = 'exp_num';
class_results{1, 2} = 'AUC';
class_results{1, 3} = 'avg_sen';
class_results{1, 4} = 'avg_spe';
class_results{1, 5} = 'F1'; 
class_results{1, 6} = 'avg_pre'; 
class_results{1, 7} = 'avg_accu';

for i = 1:exp_num
    %ratio = 0.9;
    %get training idx and training labels
    train_gene_ids = trIdx{i};                                              %get training sample indices for current round.
    %ratio
    %train_gene_ids = train_gene_ids(1:length(train_gene_ids)*ratio);
    train_labels   = curLabels(train_gene_ids, :);
    
    %get test idx and test labels
    test_gene_ids  = teIdx{i};                                              %get test sample indices for current round.
    %test_gene_ids = (1:length(test_gene_ids)*ratio);
    test_labels    = curLabels(test_gene_ids, :);

    task_cluster = get_task_cluster_km(train_labels, label_num);
    fprintf('\n>>there are %d label(s)\n', label_num);
    
        
    %building models and make predictions.
    [predicted_labels, predicted_values, task_models] = make_multi_task_model(task_cluster, ...
                                                        K(train_gene_ids, train_gene_ids),...
                                                        K(test_gene_ids, train_gene_ids),...
                                                        train_labels, test_labels);                               
    
    %evaluation
    %evals = evaluate_miml(predicted_labels(:,10), predicted_values(:,10), test_labels(:,10));
    evals = evaluate_miml(predicted_labels, predicted_values, test_labels);
    
    row_id = i+1;
    class_results{row_id, 1} = i;
    class_results{row_id, 2} = evals.AUC;
    class_results{row_id, 3} = evals.avg_sen;
    class_results{row_id, 4} = evals.avg_spe;
    class_results{row_id, 5} = evals.F1;
    class_results{row_id, 6} = evals.avg_pre;
    class_results{row_id, 7} = evals.avg_accu;
end

 

%% calculate the overall performance
performance = mean(cell2mat(class_results(2:end, 2:end)));
deviations  = std(cell2mat(class_results(2:end, 2:end)));
if size(class_results, 1) == 3
    deviations = zeros(1, size(class_results,2)-1);
end

row_id = size(class_results, 1);
class_results{row_id, 1} = 0;
for j = 2:size(class_results, 2)
    class_results{row_id, j} = [num2str(performance(j-1)), '¡À',  num2str(deviations(j-1))];
end

%% save results to file.
save(result_file_name,  'class_results',  'exp_num', 'number_of_bags');


%% restore original path
path(orginal_path);
fprintf('\n\nOK.sucessfully done.\n\n');
