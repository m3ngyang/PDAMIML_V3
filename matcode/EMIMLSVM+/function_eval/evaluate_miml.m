function evals = evaluate_miml(predicted_labels, predictedY, ground_truth)

%to avoid computing problem to add small values to predictedY
predicted_labels = full(predicted_labels);                              %if sparse, convert to full storage format
ground_truth     = full(ground_truth);                                  %if sparse, convert to full storage format

predicted_labels(predicted_labels ~= 1) = -1;                           %not necessary, but safe.
ground_truth(ground_truth ~= 1)         = -1;                           %not necessary, but safe.

predictedY       = predictedY + rand(size(predictedY))*1e-6;            %to avoid computing problem.


%calculate measures.
evals.AUC = getAUC2(predicted_labels, predictedY, ground_truth);
evals.avg_sen = avg_sensitivity(predicted_labels, ground_truth);
evals.avg_spe = avg_specificity(predicted_labels, ground_truth);
evals.F1 = getF1(predicted_labels, ground_truth);
evals.avg_pre = avg_precision(predicted_labels, ground_truth);
evals.avg_accu = avg_accuracy(predicted_labels, ground_truth);

end

