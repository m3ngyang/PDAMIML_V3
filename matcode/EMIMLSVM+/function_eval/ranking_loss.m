function rl = ranking_loss(predictedY, ground_truth)
%==================================================================================================
% get ranking_loss: rl = ranking_loss(predictedY, ground_truth)
%
% Li Yingxin, Aug. 26, 2008
%==================================================================================================
[sample_num, label_num] = size(ground_truth);

rl = 0;

for i = 1:sample_num
    label_yes = find(ground_truth(i, :)==1);
    label_no  = find(ground_truth(i, :)~=1);
    
    ly_num    = length(label_yes);
    ln_num    = length(label_no);

    invalid_num = 0;
    for j = 1:ly_num
        for k = 1:ln_num
            if predictedY(i, label_yes(j)) <= predictedY(i, label_no(k))
                invalid_num = invalid_num+1;
            end
        end
    end
    
    rl = rl + invalid_num/(ly_num*ln_num);
end
rl = rl/sample_num;
