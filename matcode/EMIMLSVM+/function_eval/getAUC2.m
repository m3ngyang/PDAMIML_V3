function auc = getAUC2(pred_lab, scores, ground_truth)
%% developed by MY 2013-06-05

label_num = size(scores, 2);
gt_num    = size(ground_truth, 2);

if (label_num ~= gt_num)
    error('mismatch between predicted and ground truth.');
end

aucs = zeros(1, label_num);

for j = 1:label_num
    y = scores(:, j);
    t = ground_truth(:, j);
    [Y,idx] = sort(-y);
    Y = -Y;
    t = t(idx);
    pidx = find(t>0);
    nidx = find(t<0);
    pins = Y(pidx); %正样本score
    nins = Y(nidx); %负样本score
    
    m = length(pins);
    n = length(nins);
    
    if(m~=0 && n~=0)
        count = 0;
        for p = 1:m
            for q = 1:n
                if(pins(p)>nins(q))
                    count = count+1;    
                end
            end
        end
        aucs(1,j) = count/(m*n);
    else
        aucs(1,j) = avg_sensitivity(pred_lab(:,j),ground_truth(:,j));%=recall?
    end
end

auc = mean(aucs(1, :));

end