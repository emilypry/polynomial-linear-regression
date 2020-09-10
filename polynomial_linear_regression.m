%  A simple polynomial linear regression example.
%  See an explanation of this program at: https://boxofcubes.wordpress.com/2020/09/04/a-simple-polynomial-regression-example/ 

clear; close all; clc

%  Load the data
load lifeexpectancy.txt
y = lifeexpectancy'

x = zeros(length(y), 1);
i=1
a=1881
while a<=2019
  x(i) = a;
  i+=1;
  a+=1;
end;
x

fprintf('\nThere is our data.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


%  Plot the data
scatter(x,y)
xlabel('Year')
ylabel('Life Expectancy')

fprintf('\nThere is our plot.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

X = [ones(length(x),1), x, sqrt(x)]

%  Mean normalization
function[X_norm, m, r] = meanNormalization(X)
X_norm = X(:, 2:size(X)(2));
m = mean(X_norm);  % m is 1xlength(X) vector with means of each column
r = range(X_norm); % r is 1xlength(X) vector with ranges of each column
X_norm = (X_norm - m) ./ r;
X_norm = [X(:, 1), X_norm];
end;

%  Cost function
function J = cost(X, y, theta)
m = size(X, 1);
predictions = X * theta;
sqrErrors = (predictions - y) .^ 2;
J = 1/(2*m) * sum(sqrErrors);
end;

%  Test cost function
theta = [0; 0; 0]
cost(X, y, theta)

fprintf('\nThere is cost of theta=[0;0;0].\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%  Gradient descent function
function [J, theta] = gradientDescent(X, y, theta, alpha, i)
m = length(y);
J = zeros(i, 1);
for a=1:i
  theta = theta - (alpha/m) * ((X*theta - y)' * X)';
  J(a) = cost(X, y, theta);
end;
end;
 
%  Normalize the data, run gradient descent (best when alpha=1 here)
X_norm = meanNormalization(X)
[j, t] = gradientDescent(X_norm, y, theta, 1, 1500)
theta = t;

fprintf('\nThere is our value for theta when gradient descent has alpha=1.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%  Plot new hypothesis
close
predictions = X_norm * theta
scatter(x, y)
hold on
plot(x, predictions)
xlabel('Year')
ylabel('Life Expectancy')

fprintf('\nThere is our hypothesis. Does not look good. Try normal equation instead of gradient descent.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%  Doesn't look great; try normal equation instead of gradient descent
close
theta_normal = pinv(X' * X) * X' * y        %  Normal equation
predictions = X * theta_normal
scatter(x,y)
hold on
plot(x, predictions)
xlabel('Year')
ylabel('Life Expectancy')

fprintf('\nThere is our hypothesis, using normal equation. Much better.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%  Much better!

%  Can demonstrate that gradient descent gets correct values for theta, when initial theta values are close to the optimal theta values
close
[j, theta_grad_des] = gradientDescent(X_norm, y, [50;-1600;1600], .1, 1500);
scatter(x,y)
hold on
plot(x, X_norm*theta_grad_des, 'r')
xlabel('Year')
ylabel('Life Expectancy')

fprintf('\nGradient descent works too, when initial values of theta are close to optimal values; had gotten stuck in local optimum previousl.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%  Works fine

%  Make predictions about life expectancy
function expectancy = predictLifeExpectancy(year)
theta = [-22649.1828911723; -11.083646156503; 1003.85125718886];
year = [1, year, sqrt(year)];
expectancy = year * theta;
end;

expectancy2020 = predictLifeExpectancy(2020)
expectancy2050 = predictLifeExpectancy(2050)

fprintf('\nCan now make predictions.\n');

