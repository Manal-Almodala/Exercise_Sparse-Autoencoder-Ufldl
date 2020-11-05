function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
%设置epsilon为1e-4
EPSILON = 1e-4;
%eye函数：返回一个size(theta,1)大小的单位矩阵 theta = [3289,1]
e_bar = eye(size(theta,1));%e_bar = [3289,3289] 单位矩阵！
%循环从1到3289，每一个存储一个变量的梯度值
for i = 1:length(theta)
    %EPSILON.*e_bar(:,i)就是给对应第i个参数加上EPSILON
    numgrad(i) = (J(theta+EPSILON.*e_bar(:,i)) - J(theta-EPSILON.*e_bar(:,i)))/(2*EPSILON);
end



%% ---------------------------------------------------------------
end