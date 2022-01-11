function H = tansigf(P,IW,Bias)
V=P*IW'; ind=ones(1,size(P,1));
BiasMatrix=Bias(ind,:);      
V=V+BiasMatrix;
H = (2 ./ (1 + exp(-2 * V)) - 1);