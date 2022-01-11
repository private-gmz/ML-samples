function Q= otho( arr )
%a^t.a=I,b^t.b=1; for compression so send iw' here not iw
% need modification
arr=arr';
[Q,S,V] = svd(arr);
S1=reshape (S',[size(S,1)*size(S,2),1]);
S=S1';
tol = max(size(arr)) * eps(max(S));
r=sum(S>tol);
Q=Q(:,1:r);

end

