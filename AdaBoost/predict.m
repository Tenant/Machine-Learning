function estimatedClass=predict(X,model)

% This function train binary-classification AdaBoost model
%
% Input:
%    X: dataset
%    model: AdaBoost model
%
% Output:
%    estimatedClass: the prediction result

num_samples=size(X,1);
estimatedClassNum=zeros(num_samples,1);

% Limit datafeaturs to original boundaries
if(length(model)>1)
    minb=model(1).boundary(1:end/2);
    maxb=model(1).boundary(end/2+1:end);
    X=bsxfun(@min,X,maxb);
    X=bsxfun(@max,X,minb);
end

% Add all results of the single weak classifiers weighted by their alpha
for t=1:length(model)
    estimatedClassNum=estimatedClassNum+model(t).alpha*ApplyClassTreshold(model(t),X);
end

estimatedClass=sign(estimatedClassNum);
end

function y = ApplyClassTreshold(h, x)
% Draw a line in one dimension (like horizontal or vertical)
% and classify everything below the line to one of the 2 classes
% and everything above the line to the other class.
if(h.direction == 1)
    y =  double(x(:,h.dimension) >= h.threshold);
else
    y =  double(x(:,h.dimension) < h.threshold);
end
y(y==0) = -1;
end