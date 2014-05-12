function [trIdx, teIdx] = gen_ind(number)
%���������ѧϰ�����Է���
%�߳�ѵ��������Ԥ��
format short;

sample_num = number;
ind = 1:1:sample_num;
time = 5;       %5��
random_ind = ind(randperm(length(ind)));
trIdx = cell(1,time);
teIdx = cell(1,time);

mid = ceil(sample_num*0.7);

for i = 1:time
    trIdx{1,i} = random_ind(:,1:mid);
    teIdx{1,i} = random_ind(:,mid+1:end);
    random_ind = [random_ind(end-mid+1:sample_num) random_ind(1:end-mid)];
end
%indexpath = [datapath,'\index.mat'];
%save(indexpath,'trIdx','teIdx');
%display('random indice done!');
end