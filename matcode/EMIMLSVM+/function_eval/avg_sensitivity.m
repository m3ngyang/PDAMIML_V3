function ase = avg_sensitivity(predicted_labels, ground_truth)
	%求标签数的sensitivity的平均值
	%sensitivity = t_pos/pos
	lab_size = size(ground_truth,2);
    ases = zeros(1,lab_size);
	
	for i = 1:lab_size
		sum2 = predicted_labels(:,i)+ ground_truth(:,i);
		sum2_ind = find(sum2==2);	%t_pos的位置和为2
		t_pos = length(sum2_ind);
		
		pos_ind = find(ground_truth(:,i)==1);
		pos = length(pos_ind);
        if pos == 0
            ases(1,i) = 0;
        else
            ases(1,i) = t_pos/pos;
        end
	end
	ase = mean(ases);
end