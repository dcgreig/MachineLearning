function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%





h=X*theta;
sum0 = sum((h - y).*(h-y));
sum1 = sum(theta.^2) - theta(1,1).^2;
disp("sum0:" + sum0/(2*m));
disp("sum1:" + sum1*lambda/(2*m));
reg = lambda/(2*m) * sum1;
disp("0:" + sum (((h-y).^2))/(2*m));
disp("1:" + lambda * (sum(theta.^2)-theta(1,1)^2)/(2*m));
%J = (1/2*m)*sum0 + reg;
J = sum0/(2*m) + sum1*lambda/(2*m);

theta_copy = theta;
theta_copy(1) = 0;
%grad = (1/m)*((predictions - y).*X) + theta_copy*lambda/m;
grad = (X'*(h - y)./m) + theta_copy*lambda/m;






% =========================================================================

grad = grad(:);

end
