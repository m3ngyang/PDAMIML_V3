function  accu  = avg_accuracy( predicted_labels, ground_truth )
%AVG_ACCURACY Summary of this function goes here
%   Detailed explanation goes here
%   accuracy = (TP+TN)/(TP+TN+FP+FN)
    lab_size = size(ground_truth,2);
    ins_num = size(ground_truth,1);
    accus = zeros(1,lab_size);
	
	for i = 1:lab_size
		sum2 = predicted_labels(:,i)+ ground_truth(:,i);
		sum2_ind = find(sum2==2);	%t_pos的位置和为2
		TN = length(sum2_ind);
		
		sum0 = predicted_labels(:,i)+ ground_truth(:,i);
		sum0_ind = find(sum0==-2);	%t_neg的位置和为-2
		TP = length(sum0_ind);
        
        accus(1,i) = (TN+TP)/ins_num;
	end
	accu = mean(accus);
end

