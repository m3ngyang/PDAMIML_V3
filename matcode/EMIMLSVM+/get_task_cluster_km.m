function task_cluster = get_task_cluster_km(train_labels,  task_num)


% Cluster the classification task into task-clusters. The task with similiar labels are put into 
% a cluster. Thus severial task-clusters can be performed 
%
% function task_cluster = get_task_cluster(train_labels,  task_num)
%
% traing_labels : n by k matrix, Rows of train_labels correspond to samples, and columns 
%                 correspond to labels (tasks).
% dist_threshold: threshold of the max distance within clusters. When the max distance with clusters is 
%                 larger than dist_threshold, then the clustering processing stop. That is the
%                 clusters must be satisfied that the max distance of each cluster must be <=
%                 dist_threshold.
%                 e.g. dist_threshold = 0.1
%                 when dist_threshold = 0, then each label is a task
%                 when dist_threshold = 1, then all labels are taken as a single task. 
%
%
% task_cluster  : m by 2 cell. 
%                 each row correspond to a task.
%                {label ids of a cluster, the average dist within a cluster, the max dist within a cluster, }
%
% <Note>
% The clustering process is redo for 20 times to get robust clustering results for demo. However, you can adjust the number of
% the repeat rounds according to your real data to get a robust results.

%sqEuclidean,correlation
%distance_type = 'correlation';                                               %distance type to perform clustering
%opts          = statset('Display','off');
%[idx, C, sumd, D] = kmeans(train_labels',  task_num, 'Distance', distance_type, 'Replicates', 20, ...
                           %'emptyaction','drop','Options', opts);

% record task clusters.
idx = 1:1:task_num;
D = Ddist(train_labels);
task_cluster = cell(task_num, 2);
for i = 1:task_num
    label_ids = find(idx == i);
    task_cluster{i, 1} = label_ids;             %the label ids.
    task_cluster{i, 2} = mean(D(label_ids, i)); %the average within distance
end

