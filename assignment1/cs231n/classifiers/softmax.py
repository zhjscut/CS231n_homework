import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    e_syi = np.exp(scores[y[i]])
    sum_e_s = np.sum(np.exp(scores))
    loss += -np.log(e_syi / sum_e_s)
    for j in range(num_classes):
      if j == y[i]:
        dW[:, y[i]] += (e_syi / sum_e_s - 1) * X[i].T
      else:
        e_sj = np.exp(scores[j])
        dW[:, j] += e_sj / sum_e_s * X[i].T
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  e_syi = np.exp(scores[np.arange(num_train), y])
  sum_e_s = np.sum(np.exp(scores), axis=1)
  loss = np.sum(-np.log(e_syi / sum_e_s))
  sum_e_s = sum_e_s[:, np.newaxis]
  margins = np.exp(scores) / sum_e_s   #不知道应该叫什么名字,就随便取了个跟SVM一样的名字
  margins[np.arange(num_train), y] -= 1 #对比naive中的if分支两种情况,可以看到它们在形式上只差了个1
  dW = X.T.dot(margins)

  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W          
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

