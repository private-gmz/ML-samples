%%% calculating narnet initial input weights using ELM autoencoder %%%%%%%%%%

function [Bias,TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = newelm1(minii,smoothdata, nHiddenNeurons, ActivationFunction, Block_Range)
       
nInputNeurons=Block_Range;

Bias =randn(1,nHiddenNeurons);
bbb=otho(Bias);
Bias=bbb';
IW = randn(nHiddenNeurons,nInputNeurons);
iiww=otho(IW);
IW=iiww';
beta = zeros(nHiddenNeurons,nInputNeurons);

M = inv(0.00001*eye(nHiddenNeurons));
for i=1:size(smoothdata,2)-1
while 1 rr=randi([1,size(smoothdata,2)],1,1);if (rr~=i)&&(rr~=i+1) break;end;end;
T=smoothdata{i}(1:minii,:); P=smoothdata{i}(1:minii,:);
TV.T=smoothdata{i+1}(1:minii,:); TV.P=smoothdata{i+1}(1:minii,:);
TV.T2=smoothdata{rr}(1:minii,:); TV.P2=smoothdata{rr}(1:minii,:);
T=removeconstantrows(T);
T=mapminmax(T);
P=removeconstantrows(P);
P=mapminmax(P);
TV.T=removeconstantrows(TV.T);
TV.T=mapminmax(TV.T);
TV.P=removeconstantrows(TV.P);
TV.P=mapminmax(TV.P);
TV.T2=removeconstantrows(TV.T2);
TV.T2=mapminmax(TV.T2);
TV.P2=removeconstantrows(TV.P2);
TV.P2=mapminmax(TV.P2);
T=reshape (T',[size(T,1)*size(T,2),1]);
P=reshape (P',[size(P,1)*size(P,2),1]);
TV.T=reshape (TV.T',[size(TV.T,1)*size(TV.T,2),1]);
TV.P=reshape (TV.P',[size(TV.P,1)*size(TV.P,2),1]);
TV.T2=reshape (TV.T2',[size(TV.T2,1)*size(TV.T2,2),1]);
TV.P2=reshape (TV.P2',[size(TV.P2,1)*size(TV.P2,2),1]);
ss=1;
for ii=1:round((minii-(Block_Range-1))/2)
T1(ii,1:Block_Range)=T(ss:ss+(Block_Range-1));
P1(ii,1:Block_Range)=P(ss:ss+(Block_Range-1));
TV.P1(ii,1:Block_Range)=TV.P(ss:ss+(Block_Range-1));
TV.T1(ii,1:Block_Range)=TV.T(ss:ss+(Block_Range-1));
TV.P22(ii,1:Block_Range)=TV.P2(ss:ss+(Block_Range-1));
TV.T22(ii,1:Block_Range)=TV.T2(ss:ss+(Block_Range-1));
ss=ss+2;
end
T=T1;P=P1;TV.T=TV.T1;TV.P=TV.P1;TV.T2=TV.T22;TV.P2=TV.P22;
nTrainingData=size(P,1); 
nTestingData=size(TV.P,1);
nInputNeurons=size(P,2);

start_time_train=cputime;
n = 1;

while n <= nTrainingData
    
    Block = randi(Block_Range,1,1);    
    if (n+Block-1) > nTrainingData
        Pn = P(n:nTrainingData,:);    Tn = T(n:nTrainingData,:);
        Block = size(Pn,1);             %%%% correct the block size
    else
        Pn = P(n:(n+Block-1),:);    Tn = T(n:(n+Block-1),:);
    end 
    switch lower(ActivationFunction)
        case{'rbf'}
            H = radbas(Pn,IW,Bias);
        case{'sig'}
            H = SigActFun(Pn,IW,Bias);
        case{'tansigf'}
            H = tansigf(Pn,IW,Bias);
        case{'hardlim'}
            H = HardlimActFun(Pn,IW,Bias);
    end    

    scale=1/0.999;
    M = (scale*M) - (scale*M) * H' * (eye(Block) + H * (scale*M) * H')^(-1) * H * (scale*M); 
    beta = (0.999*beta) + M * H' * (Tn - H * (0.999*beta));
    n = n + Block;
end
%% beta' for IW

end_time_train=cputime;
TrainingTime{i}=end_time_train-start_time_train;        
clear Pn Tn H;
switch lower(ActivationFunction)
    case{'rbf'}
        HTrain = radbas(P, IW, Bias);
    case{'sig'}
        HTrain = SigActFun(P, IW, Bias);
    case{'tansigf'}
        HTrain = tansigf(P, IW, Bias);
    case{'hardlim'}
        HTrain = HardlimActFun(P, IW, Bias);
end
Y=HTrain * beta;


%%%%%%%%%%% Performance Evaluation
start_time_test=cputime; 
%%1
switch lower(ActivationFunction)
    case{'rbf'}
        HTest = radbas(TV.P, IW, Bias);
    case{'sig'}
        HTest = SigActFun(TV.P, IW, Bias);
    case{'tansigf'}
        HTest = tansigf(TV.P, IW, Bias);
    case{'hardlim'}
        HTest = HardlimActFun(TV.P, IW, Bias);
end    
TY=HTest * beta;
clear HTest;
%%25
switch lower(ActivationFunction)
    case{'rbf'}
        HTest2 = radbas(TV.P2, IW, Bias);
    case{'sig'}
        HTest2 = SigActFun(TV.P2, IW, Bias);
    case{'tansigf'}
        HTest2 = tansigf(TV.P2, IW, Bias);
    case{'hardlim'}
        HTest2 = HardlimActFun(TV.P2, IW, Bias);
end    
TY2=HTest2 * beta;
clear HTest2;
end_time_test=cputime;
TestingTime{i}=end_time_test-start_time_test;

TrainingAccuracy{i}=sqrt(mse(T - Y));               
    TestingAccuracy{i}=(sqrt(mse(TV.T - TY))+sqrt(mse(TV.T2 - TY2)))/2;
	
end
dlmwrite(['weights\\B.mat'],beta);
