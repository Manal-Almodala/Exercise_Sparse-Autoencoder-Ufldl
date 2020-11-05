function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64)  输入层的节点数目为64
% hiddenSize: the number of hidden units (probably 25)  隐含层的节点数目为25个
% lambda: weight decay parameter   权值衰减参数
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks
%                           like a lower-case "p"). 期望平均激活rho
% beta: weight of sparsity penalty term 稀疏惩罚
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is
% the i-th training example.  数据集 格式为64*10000 第(:,i)为第i个数据
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 


a1 = data;%输入层(第一层)的数据
%repmat(b1,1,size(data,2))作用在于将b1变为25*10000的数据
%W1大小为25*64 data大小为64*10000 相乘后为25 * 10000 需要加上 b1
%repmat(b1,1,size(data,2))的作用在于将b1扩充为25*10000
a2 = sigmoid(W1*data + repmat(b1,1,size(data,2))); %隐含层(第二层)的输出
a3 = sigmoid(W2*a2 + repmat(b2,1,size(data,2))); %输出层(第三层)的输出

% Cost损失函数
%(1/size(data,2))为1/m
%sum(sum(0.5*(a3 - data).^2))为taget和输出a3之间的误差
%((lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2)))是weight decay 
J = ((1/size(data,2))*sum(sum(0.5*(a3 - data).^2))) + ((lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2))));
%求^rho_j
%(1/size(data,2))为1/m
% sum(a2,2) 为对a2进行求和
rho_approx = (1/size(data,2))*sum(a2,2);
%求KL散度，根据第4节中的KL散度公式
KL = sum(sparsityParam.*log(sparsityParam./rho_approx) + (1 - sparsityParam).*...
    log((1-sparsityParam)./(1-rho_approx)));
%总损失函数为基本损失函数加KL散度
cost = J + (beta * KL);

% 反向传播过程 
del3 = -(data - a3).*(a3.*(1-a3)); %求del3
del2 = ((W2'*del3) + repmat(beta.*((-sparsityParam./rho_approx)+...
    ((1-sparsityParam)./(1-rho_approx))),1,size(data,2))).*(a2.*(1-a2));%求del2
%求每个权重所需改变的梯度
W2grad = ((del3*a2')/size(data,2)) + (lambda*W2);
b2grad = sum(del3,2)/size(data,2);
W1grad = ((del2*a1')/size(data,2)) + (lambda*W1);
b1grad = sum(del2,2)/size(data,2);





%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

