load data.mat
X=data(:,1:56);
y=data(:,57);
y=2*y-3;

% Train model
[estimatedClass, model]=train(X,y,1000);

% Predict
y_hat=predict(X,model);

% Calculate accuracy
accu=sum(y_hat==y)/length(y);
info=sprintf("The accuracy: %f\n",accu);
fprintf(info);