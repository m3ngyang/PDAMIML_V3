function ap = avg_precious(predictedY, ground_truth)
%==================================================================================================
% ap = avg_precious(predictedY, ground_truth)
%
% Li Yingxin, Aug. 26. 2008
% =================================================================================================

sample_num = size(ground_truth, 1);

ap = 0;
for i = 1:sample_num
    label_yes = find(ground_truth(i, :)==1);
    
    ly_num = length(label_yes);
    sorted = sort(predictedY(i,:), 'descend');

    pvalue = 0;
    for j = 1:ly_num
        jorder = find(sorted == predictedY(i, label_yes(j)));
        
        pnum = 0;
        for k = 1:ly_num
            korder = find(sorted == predictedY(i, label_yes(k)));
            
            if korder <= jorder
                pnum = pnum + 1;
            end
        end
        
        pvalue = pvalue + pnum/jorder;
    end
    
    ap  = ap + pvalue/ly_num;
end

ap = ap/sample_num;
