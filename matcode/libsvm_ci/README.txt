We modified the Libsvm ver.2.89 to:

(1) model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options', cs);
    the parameter cs is the vector of penalty factors, each for a sample.
    this alteration is just for C-SVC.
(2) cross validation: there are two outputs, the first is the conventional cv_value, and the second one is the predicted result for further exploiting.



LIBSVM is a library for support vector
machines (http://www.csie.ntu.edu.tw/~cjlin/libsvm). It is very easy to use as
the usage and the way of specifying parameters are the same as that of LIBSVM.
