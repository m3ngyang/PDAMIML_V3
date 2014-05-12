function ape = avg_specificity(predicted_labels, ground_truth)
	%求标签数的specificity的平均值
	%specificity = t_neg/neg
	lab_size = size(ground_truth,2);
    apes = zeros(1,lab_size);
	
	for i = 1:lab_size
		sum0 = predicted_labels(:,i)+ ground_truth(:,i);
		sum0_ind = find(sum0==-2);	%t_neg的位置和为-2
		t_neg = length(sum0_ind);
		
		neg_ind = find(ground_truth(:,i)==-1);
		neg = length(neg_ind);
        if neg == 0
            apes(1,i) = 0;
        else
            apes(1,i) = t_neg/neg;
        end
	end
	ape = mean(apes);
end