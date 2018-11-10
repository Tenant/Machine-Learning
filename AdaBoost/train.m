function [estimatedClassAdaBoost, model]=train(X,y,num_iter)

% This function train binary-classification AdaBoost model
%
% Input:
%    x: size(number_samples, number_features)
%    y: size(number_samples, 1)
%    num_iter: number of iterations
%
% Output:
%    estimatedClass: y_hat
%    model: model parameters

model=struct;
num_samples=length(y);

% Initialize weight of training samples equivalently
w=ones(num_samples,1)/num_samples;

estimatedClassNum=zeros(num_samples,1);

% Calculate maxium and minimun of the data
boundary=[min(X,[],1) max(X,[],1)];


% Training
for iter=1:num_iter
    
    % Find the best threshold to seperate the data in two classes
    [estimatedClass,err,h]=WeightedThresholdClassifier(X,y,w);
    
    % Weak classifier influence on total result is basesd on the current
    % classification error
    alpha= 1/2 * log((1-err)/max(err,eps));
    
    % Store the model parameters
    model(iter).alpha=alpha;
    model(iter).dimension=h.dimension;
    model(iter).threshold=h.threshold;
    model(iter).direction=h.direction;
    model(iter).boundary=boundary;
    
    % Update w so that these wrongly classified samples will gain more
    % weights
    w=w.*exp(-model(iter).alpha.*y.*estimatedClass);
    w=w./sum(w);
    
    % Calculate the current error of the cascade of weak
    estimatedClassNum=estimatedClassNum+y*model(iter).alpha;
    estimatedClassAdaBoost=sign(estimatedClassNum);
    model(iter).error=sum(estimatedClassAdaBoost~=y)/num_samples;
end
end

function [estimateclass,err,h] = WeightedThresholdClassifier(datafeatures,dataclass,dataweight)

% Number of treshold steps
ntre=2e5;

% Split the data in two classes 1 and -1
r1=datafeatures(dataclass<0,:); w1=dataweight(dataclass<0);
r2=datafeatures(dataclass>0,:); w2=dataweight(dataclass>0);

% Calculate the min and max for every dimensions
minr=min(datafeatures,[],1)-1e-10; maxr=max(datafeatures,[],1)+1e-10;

% Make a weighted histogram of the two classes
p2c= ceil((bsxfun(@rdivide,bsxfun(@minus,r2,minr),(maxr-minr)))*(ntre-1)+1+1e-9);   p2c(p2c>ntre)=ntre;
p1f=floor((bsxfun(@rdivide,bsxfun(@minus,r1,minr),(maxr-minr)))*(ntre-1)+1-1e-9);  p1f(p1f<1)=1;
ndims=size(datafeatures,2);
i1=repmat(1:ndims,size(p1f,1),1);  i2=repmat(1:ndims,size(p2c,1),1);
h1f=accumarray([p1f(:) i1(:)],repmat(w1(:),ndims,1),[ntre ndims],[],0);
h2c=accumarray([p2c(:) i2(:)],repmat(w2(:),ndims,1),[ntre ndims],[],0);

% This function calculates the error for every all possible treshold value
% and dimension
h2ic=cumsum(h2c,1);
h1rf=cumsum(h1f(end:-1:1,:),1); h1rf=h1rf(end:-1:1,:);
e1a=h1rf+h2ic;
e2a=sum(dataweight)-e1a;

% We want the treshold value and dimension with the minimum error
[err1a,ind1a]=min(e1a,[],1);  dim1a=(1:ndims); dir1a=ones(1,ndims);
[err2a,ind2a]=min(e2a,[],1);  dim2a=(1:ndims); dir2a=-ones(1,ndims);
A=[err1a(:),dim1a(:),dir1a(:),ind1a(:);err2a(:),dim2a(:),dir2a(:),ind2a(:)];
[err,i]=min(A(:,1)); dim=A(i,2); dir=A(i,3); ind=A(i,4);
thresholds = linspace(minr(dim),maxr(dim),ntre);
thr=thresholds(ind);

% Apply the new treshold
h.dimension = dim; 
h.threshold = thr; 
h.direction = dir;
estimateclass=ApplyClassTreshold(h,datafeatures);
end

function y = ApplyClassTreshold(h, x)

if(h.direction == 1)
    y =  double(x(:,h.dimension) >= h.threshold);
else
    y =  double(x(:,h.dimension) < h.threshold);
end
y(y==0) = -1;
end