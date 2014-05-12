function [ pre ] = avg_precision( predicted_labels, ground_truth )
%AVG_PRECISION Summary of this function goes here
%   Detailed explanation goes here
%   precison = TP/(TP+FP)
    lab_size = size(ground_truth,2);
    pres = zeros(1,lab_size);
    
    for i = 1:lab_size
        sum2 = predicted_labels(:,i)+ ground_truth(:,i);
		sum2_ind = find(sum2==2);	%t_pos的位置和为2
		t_pos = length(sum2_ind);
		
		pos_ind = find(predicted_labels(:,i)==1);
		pos = length(pos_ind);
        if pos == 0
            pres(1,i) = 0;
        else
            pres(1,i) = t_pos/pos;
        end
    end
    
    pre = mean(pres);
end

