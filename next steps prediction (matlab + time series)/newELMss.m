%%%%%% this function calculate output weights of ELM neural networks using input weights calculated using ELM auto encoder in newelm1 function %%%%%%%%%%%%%%%%

load dataSRnew;
minii=min(cellfun('size',dataSRnew,1));
range=delay*2;
[Bias,TrainingAccuracy, TestingAccuracy]= newelm1(minii,dataSRnew, hiddennum, 'tansigf', range);
[C,I] = min(cell2mat(TestingAccuracy))
%%%% ELM autoencoder weights
wi= dlmread(['weights\\B.mat']);
x=dataSRnew{I}(1:minii,:);
x=removeconstantrows(x);
x=mapminmax(x);
x=reshape (x',[size(x,1)*size(x,2),1]);
ss=1;
for ii=1:round((minii-(range-1))/2)-1
z(ii,1:(range))=x(ss:ss+(range-1));
ss=ss+2;
end
%%%%
HH=tansigf(z, wi, Bias);
%%%%
sss=13;
for ii=1:round((minii-(range-1))/2)-1
t(ii,1:2)=x(sss:sss+1);
sss=sss+2;
end

%%%% output weights
pp=pinv(HH)*t;
