from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        h1, cache1 = affine_relu_forward(X, W1, b1) #这个cache里保存的是(x, W1, b1, x),矩阵乘和relu的输入
        scores, cache2 = affine_forward(h1, W2, b2) #这个cache里保持的是(h1, W2, b2),网络第2层的输入
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y) #最后的score其实是score_softmax,所以才会有dscore这一说,即score还不是最后一层
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        dh1, dW2, db2 = affine_backward(dscores, cache2)
        dx, dW1, db1 = affine_relu_backward(dh1, cache1)
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        if self.normalization is None:
            for i in range(0, self.num_layers):
                if i == self.num_layers-1:
                    self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], num_classes)
                    self.params['b'+str(i+1)] = np.zeros(num_classes)     
                elif i == 0:
                    self.params['W'+str(i+1)] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
                    self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
                else:
                    self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
                    self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])          
        elif self.normalization == 'batchnorm' or self.normalization == 'layernorm': 
            for i in range(0, self.num_layers):
                if i == self.num_layers-1:
                    self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], num_classes)
                    self.params['b'+str(i+1)] = np.zeros(num_classes)     
                elif i == 0:
                    self.params['W'+str(i+1)] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
                    self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
                    self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])
                else:
                    self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
                    self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])          
                    self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])                                     
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        caches, hs = {}, {}
        if self.normalization == None:
            for i in range(0, self.num_layers):
                if i == self.num_layers-1:
                    scores, caches['cache'+str(i+1)] = affine_forward(hs['h'+str(i)], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                elif i == 0:
                    hs['h'+str(i+1)], caches['cache'+str(i+1)] = affine_relu_forward_new(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.dropout_param)
                else:
                    hs['h'+str(i+1)], caches['cache'+str(i+1)] = affine_relu_forward_new(hs['h'+str(i)], self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.dropout_param)
        elif self.normalization == 'batchnorm' or self.normalization == 'layernorm':      
            for i in range(0, self.num_layers):
                if i == self.num_layers-1:
                    scores, caches['cache'+str(i+1)] = affine_forward(hs['h'+str(i)], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                elif i == 0:
                    hs['h'+str(i+1)], caches['cache'+str(i+1)] = affine_relu_norm_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i], self.dropout_param)
                else:
                    hs['h'+str(i+1)], caches['cache'+str(i+1)] = affine_relu_norm_forward(hs['h'+str(i)], self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i], self.dropout_param)
                 
        # 下面的语句仅供对比排错
        # W1, b1 = self.params['W1'], self.params['b1'] #现在还不能实现对可变层数的处理,还是固定2层
        # W2, b2 = self.params['W2'], self.params['b2']
        # W3, b3 = self.params['W3'], self.params['b3']
        # h1, cache1 = affine_relu_forward(X, W1, b1) #这个cache里保存的是(x, W1, b1, x),矩阵乘和relu的输入
        # h2, cache2 = affine_relu_forward(h1, W2, b2)
        # scores, cache3 = affine_forward(h2, W3, b3) #这个cache里保持的是(h1, W2, b2),网络第2层的输入
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y) #最后的score其实是score_softmax,所以才会有dscore这一说,即score还不是最后一层
        dWs, dbs, dhs, dgammas, dbetas = {}, {}, {}, {}, {}
        if self.normalization == None:
            for i in range(self.num_layers-1, -1, -1):
                loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)] ** 2)
                if i == self.num_layers-1:
                    dhs['h'+str(i)], dWs['W'+str(i+1)], dbs['b'+str(i+1)] = affine_backward(dscores, caches['cache'+str(i+1)])
                elif i == 0:
                    dx , dWs['W'+str(i+1)], dbs['b'+str(i+1)] = affine_relu_backward_new(dhs['h'+str(i+1)], caches['cache'+str(i+1)])
                else:
                    dhs['h'+str(i)], dWs['W'+str(i+1)], dbs['b'+str(i+1)] = affine_relu_backward_new(dhs['h'+str(i+1)], caches['cache'+str(i+1)])
                
                grads['W'+str(i+1)] = dWs['W'+str(i+1)] + self.reg * self.params['W'+str(i+1)]
                grads['b'+str(i+1)] = dbs['b'+str(i+1)]
        elif self.normalization == 'batchnorm' or self.normalization == 'layernorm':
            for i in range(self.num_layers-1, -1, -1):
                loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)] ** 2)
                if i == self.num_layers-1:
                    dhs['h'+str(i)], dWs['W'+str(i+1)], dbs['b'+str(i+1)] = affine_backward(dscores, caches['cache'+str(i+1)])
                elif i == 0:
                    dx , dWs['W'+str(i+1)], dbs['b'+str(i+1)], dgammas['gamma'+str(i+1)], dbetas['beta'+str(i+1)] = affine_relu_norm_backward(dhs['h'+str(i+1)], caches['cache'+str(i+1)], self.normalization)
                else:
                    dhs['h'+str(i)], dWs['W'+str(i+1)], dbs['b'+str(i+1)], dgammas['gamma'+str(i+1)], dbetas['beta'+str(i+1)] = affine_relu_norm_backward(dhs['h'+str(i+1)], caches['cache'+str(i+1)], self.normalization)
                
                grads['W'+str(i+1)] = dWs['W'+str(i+1)] + self.reg * self.params['W'+str(i+1)]
                grads['b'+str(i+1)] = dbs['b'+str(i+1)]    
                if i != self.num_layers - 1:    
                    grads['gamma'+str(i+1)] = dgammas['gamma'+str(i+1)]        
                    grads['beta'+str(i+1)] = dbetas['beta'+str(i+1)]        
        # 下面的语句仅供对比排错
        # loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        # dh2, dW3, db3 = affine_backward(dscores, cache3)
        # dh1, dW2, db2 = affine_relu_backward(dh2, cache2)
        # dx, dW1, db1 = affine_relu_backward(dh1, cache1)
        # dW1 += self.reg * W1
        # dW2 += self.reg * W2
        # dW3 += self.reg * W3
        # grads['W1'] = dW1
        # grads['b1'] = db1
        # grads['W2'] = dW2
        # grads['b2'] = db2   
        # grads['W3'] = dW3
        # grads['b3'] = db3   
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_relu_norm_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a1, fc_cache = affine_forward(x, w, b)
    if bn_param == {}: #如果是层归一化,bn_param会被设置成{} (不做归一化时则没有bn_param这个参数)
        a2, bn_cache = layernorm_forward(a1, gamma, beta, bn_param) #这时bn_param(也即ln_param没有携带任何信息,eps会使用默认值1e-5)
    else:
        a2, bn_cache = batchnorm_forward(a1, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a2)
    if dropout_param != {}:
        out, drop_cache = dropout_forward(out, dropout_param)
        cache = (fc_cache, relu_cache, relu_cache, drop_cache)
    else:
        cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_relu_norm_backward(dout, cache, mode='batchnorm'):
    """
    Backward pass for the affine-relu convenience layer
    """
    if len(cache) == 3:
        fc_cache, bn_cache, relu_cache = cache
        da2 = relu_backward(dout, relu_cache)
        if mode == 'batchnorm':
            da1, dgamma, dbeta = batchnorm_backward(da2, bn_cache)
        elif mode == 'layernorm':
            da1, dgamma, dbeta = layernorm_backward(da2, bn_cache)
        dx, dw, db = affine_backward(da1, fc_cache)
    elif len(cache) == 4:
        fc_cache, bn_cache, relu_cache, drop_cache = cache
        dout = dropout_backward(dout, drop_cache)
        da2 = relu_backward(dout, relu_cache)
        if mode == 'batchnorm':
            da1, dgamma, dbeta = batchnorm_backward(da2, bn_cache)
        elif mode == 'layernorm':
            da1, dgamma, dbeta = layernorm_backward(da2, bn_cache)
        dx, dw, db = affine_backward(da1, fc_cache)        
    return dx, dw, db, dgamma, dbeta


def affine_relu_forward_new(x, w, b, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    if dropout_param != {}:
        out, drop_cache = dropout_forward(out, dropout_param)
        cache = (fc_cache, relu_cache, drop_cache)
    else:
        cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward_new(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    if len(cache) == 2:
        fc_cache, relu_cache = cache
        da = relu_backward(dout, relu_cache)
        dx, dw, db = affine_backward(da, fc_cache)
    elif len(cache) == 3:
        fc_cache, relu_cache, drop_cache = cache
        dout = dropout_backward(dout, drop_cache)
        da = relu_backward(dout, relu_cache)
        dx, dw, db = affine_backward(da, fc_cache)        
    return dx, dw, db    