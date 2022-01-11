%%%%%%%%%%%%%%%%%%% what is this? %%%%%%%%%%%%%%%%%%%
% in this file I train and test narnet neural network on x,y coordinates generated 
% using bonnmotion (motion generator), initial weights calculated using ELM and ELM autoencoder

%%%%%%%%%%%%%%% notes for me %%%%%%%%%%%%%%%%%%%%%%%  
%with 11 perf = 1.0450e-04 perfc =0.1831
%with 9 perf = 1.0459e-04 perfc =0.1808
%with 6 perf =   1.0510e-04 perfc =0.1759
%0.1800 0.23 0.2082
% with elm_ae 0.1605 #30
% with #35 0.1870
% #40 with 0.1589
%best 45

%%%% bonnmotion data is loaded and saved in dataSRssnew matrix
load dataSRssnew;
    start_time_train=cputime;
   
%%%%%% from trying     
hiddennum=10;

%%%%% best delay is chosen using autocorrelation
%all 98 contains 6 and only 57 contain 10
delay=6;


%%%%%% building the model
net = narnet(1:delay,hiddennum);
[Xs,Xi,Ai,Ts] = preparets(net,{},{},dataSRssnew);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.performParam.normalization = 'standard';
%%%%%%%%%%%%%
net = configure(net, Xs, Ts );


%%%%%%% calculate initial weights
run newELMss;
bb=rand(2,1)*sqrt(2);
%net.initFcn ='initlay';
%net.inputWeights{1,1}.initFcn='';
%net.inputWeights{2,1}.initFcn='';

%net.layerWeights{1,1}.initFcn='';
%net.layerWeights{1,2}.initFcn='';
%net.layerWeights{2,1}.initFcn='';
%net.layerWeights{2,2}.initFcn='';

%net.biases{2,1}.initFcn='';
%net.biases{1,1}.initFcn='';
net.IW{1,1}=wi*sqrt(2/(delay*2));
net.LW{2,1}=pp'*sqrt(2/hiddennum);
net.b{1}=Bias'*sqrt(2);
net.b{2}=bb;
wb1=net.IW;
%net.trainFcn='trainbr';
net.trainParam.epochs =500;
%%net.trainParam.goal=1.04e-04;11
%%net.trainParam.goal=1.035e-04;0.1857
net.divideFcn='divideblock'; % I did not try randomizing these states here yet, possible future improvement...
 
%%%%%%%%%%%%%%%%
%%% train the model 
rng(0)
[net,tr] = train(net,Xs,Ts,Xi,Ai);
wb2=net.IW;
Y = net(Xs,Xi);
perf10r0_500 = perform(net,Ts,Y)


%% remove delay (for one step prediction) and test on new data
load dataSRss11
nets = removedelay(net);
[xs,xis,ais,ts] = preparets(nets,{},{},dataSRss);
ys = nets(xs,xis,ais);
newstepAheadPerformance10r0_500= perform(nets,ts,ys)

%test=load_data1('C:\Users\zeinab\Desktop\outputr.csv');
%test = tonndata(test,false,false);
%height_net = nets;
%[xs,xis,ais,ts] = preparets(height_net,{},{},test);
%y1= height_net(xs,xis,ais);

%%%% close net for multi-steps prediction
 netc = closeloop(net);
[xc,xic,aic,tc] = preparets(netc,{},{},dataSRss);
[yc,Xcf,Acf] = netc(xc,xic,aic);
newperfc10r0_500= perform(netc,tc,yc)
%12=>0.2353,13=>0.2131
%12=>0.2353,17=>0.1876
end_time_train=cputime;
time10r0=end_time_train-start_time_train;