"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
from eecs598 import Solver
from a3_helper import svm_loss, softmax_loss
from fully_connected_networks import *

def hello_convolutional_networks():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from convolutional_networks.py!')


class Conv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
      
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ##############################################################################
    # TODO: Implement the convolutional forward pass.                            #
    # Hint: you can use the function torch.nn.functional.pad for padding.        #
    # Note that you are NOT allowed to use anything in torch.nn in other places. #
    ##############################################################################
    # Replace "pass" statement with your code

    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = torch.zeros((N, F, H_out, W_out), device=x.device, dtype=x.dtype)
    # padding ref: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.pad
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))

    for i in range(H_out):
      h_s = i * stride
      h_e = h_s + HH
      for j in range(W_out):
        w_s = j * stride
        w_e = w_s + WW
        for k in range(F):
          out[:, k, i, j] = torch.sum(x_pad[:, :, h_s:h_e, w_s:w_e] * w[k], dim=(1,2,3)) + b[k]

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.: (N, F, H_out, W_out)
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x: (N, C, H, W)
    - dw: Gradient with respect to w: (F, C, HH, WW)
    - db: Gradient with respect to b: (F,)
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # Replace "pass" statement with your code
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    # ref: https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
    # ref2: https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509

    # dw
    dw = torch.zeros((*w.shape), device=w.device, dtype=w.dtype)

    # dx
    dx = torch.zeros((*x.shape), device=x.device, dtype=x.dtype)
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    dx_pad = torch.zeros((*x_pad.shape), device=dx.device, dtype=dx.dtype)

    for i in range(H_out):
      h_s = i * stride
      h_e = h_s + HH
      for j in range(W_out):
        w_s = j * stride
        w_e = w_s + WW
        for k in range(F):
          tmp = dout[:, k, i, j].reshape(dout[:, k, i, j].shape[0], 1, 1, 1) * x_pad[:, :, h_s:h_e, w_s:w_e]
          dw[k] += torch.sum(tmp, dim=0)
        for l in range(N):
          tmp = dout[l, :, i, j].reshape(dout[l, :, i, j].shape[0], 1, 1, 1) * w
          dx_pad[l, :, h_s:h_e, w_s:w_e] += torch.sum(tmp, dim=0)

    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    # db
    db = torch.sum(dout, dim=(0,2,3))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx, dw, db


class MaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here.

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max-pooling forward pass                              #
    #############################################################################
    # Replace "pass" statement with your code

    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    out = torch.zeros((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    for i in range(H_out):
      h_s = i * pool_height
      h_e = h_s + stride
      for j in range(W_out):
        w_s = j * pool_width
        w_e = w_s + stride
        out[:, :, i, j] = torch.max(x[:, :, h_s:h_e, w_s:w_e].reshape(x.shape[0], x.shape[1], -1), dim=2)[0]

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, pool_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max-pooling backward pass                             #
    #############################################################################
    # Replace "pass" statement with your code

    (x, pool_param) = cache
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    dx = torch.zeros((*x.shape), device=x.device, dtype=x.dtype)

    for i in range(H_out):
      h_s = i * pool_height
      h_e = h_s + stride
      for j in range(W_out):
        w_s = j * pool_width
        w_e = w_s + stride
        for k in range(N):
          for l in range(C):
            tmp = x[k, l, h_s:h_e, w_s:w_e]
            mask = (tmp == tmp.max()).double()
            mask *= dout[k, l, i, j]
            dx[k, l, h_s:h_e, w_s:w_e] = mask

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  conv - relu - 2x2 max pool - linear - relu - linear - softmax
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7,
         hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
         dtype=torch.float64, device='cpu'):
    """
    Initialize a new network.
    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Width/height of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian centered at 0.0   #
    # with standard deviation equal to weight_scale; biases should be          #
    # initialized to zero. All weights and biases should be stored in the      #
    #  dictionary self.params. Store weights and biases for the convolutional  #
    # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
    # weights and biases of the hidden linear layer, and keys 'W3' and 'b3'    #
    # for the weights and biases of the output linear layer.                   #
    #                                                                          #
    # IMPORTANT: For this assignment, you can assume that the padding          #
    # and stride of the first convolutional layer are chosen so that           #
    # **the width and height of the input are preserved**. Take a look at      #
    # the start of the loss() function to see how that happens.                #               
    ############################################################################
    # Replace "pass" statement with your code

    self.params['W1'] = torch.normal(0.0, weight_scale, size=(num_filters, input_dims[0], filter_size, filter_size)).to(dtype).to(device)
    self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=device)

    in_dim = num_filters*input_dims[-1]**2 // 4

    self.params['W2'] = torch.normal(0.0, weight_scale, size=(in_dim, hidden_dim)).to(dtype).to(device)
    self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)

    self.params['W3'] = torch.normal(0.0, weight_scale, size=(hidden_dim, num_classes)).to(dtype).to(device)
    self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = checkpoint['dtype']
    self.reg = checkpoint['reg']
    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    Input / output: Same API as TwoLayerNet.
    """
    X = X.to(self.dtype)
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # Remember you can use the functions defined in your implementation above. #
    ############################################################################
    # Replace "pass" statement with your code
    # conv - relu - 2x2 max pool - linear - relu - linear - softmax

    outs = []
    caches = []

    ## MY IMPLEMENTATION --> WORKS FINE BUT SUPER SLOW
    # out, cache = Conv.forward(X, W1, b1, conv_param)
    # outs.append(out)
    # caches.append(cache)

    # out, cache = ReLU.forward(out)
    # outs.append(out)
    # caches.append(cache)

    # out, cache = MaxPool.forward(out, pool_param)
    # outs.append(out)
    # caches.append(cache)

    out, cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
    outs.append(out)
    caches.append(cache)

    out, cache = Linear_ReLU.forward(out, W2, b2)
    outs.append(out)
    caches.append(cache)

    scores, cache = Linear.forward(out, W3, b3)
    outs.append(scores)
    caches.append(cache)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code

    loss, da = softmax_loss(outs.pop(), y)

    if self.reg > 0:
      loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2) + torch.sum(W3 * W3))

    ds = {}

    ds['X3'], grads['W3'], grads['b3'] = Linear.backward(da, caches.pop())
    ds['X2'], grads['W2'], grads['b2'] = Linear_ReLU.backward(ds['X3'], caches.pop())

    ## MY IMPLEMENTATION --> WORKS FINE BUT SUPER SLOW
    # ds['X2_pool'] = MaxPool.backward(ds['X2'], caches.pop())
    # ds['X2_relu'] = ReLU.backward(ds['X2_pool'], caches.pop())
    # _, grads['W1'], grads['b1']  = Conv.backward(ds['X2_relu'], caches.pop())
    _, grads['W1'], grads['b1']  = Conv_ReLU_Pool.backward(ds['X2'], caches.pop())

    grads['W1'] += 2 * self.reg * W1
    grads['W2'] += 2 * self.reg * W2
    grads['W3'] += 2 * self.reg * W3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dims=(3, 32, 32),
               num_filters=[8, 8, 8, 8, 8],
               max_pools=[0, 1, 2, 3, 4],
               batchnorm=False,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               weight_initializer=None,
               dtype=torch.float64, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights, or the string "kaiming" to use Kaiming initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
  
    if device == 'cuda':
      device = 'cuda:0'
    
    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.params.                                                  #
    #                                                                          #
    # Weights for conv and fully-connected layers should be initialized        #
    # according to weight_scale. Biases should be initialized to zero.         #
    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
    # to ones and zeros respectively.                                          #           
    ############################################################################
    # Replace "pass" statement with your code

    # {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    for i in range(self.num_layers):
      if weight_scale == 'kaiming':
        if i == 0:  # first conv
          self.params[f'W{i}'] = kaiming_initializer(input_dims[0], num_filters[i], K=3, device=device, dtype=dtype)
          if self.batchnorm:
            self.params[f'gamma{i}'] = torch.ones(num_filters[i], device=device, dtype=dtype)
            self.params[f'beta{i}'] = torch.zeros(num_filters[i], device=device, dtype=dtype)
        elif i == self.num_layers - 1:  # linear layer
          in_dim = num_filters[-1] * ((input_dims[-1] // (2 ** len(max_pools))) ** 2)
          self.params[f'W{i}'] = kaiming_initializer(in_dim, num_classes, K=None, device=device, dtype=dtype)
        else:  # conv
          self.params[f'W{i}'] = kaiming_initializer(num_filters[i-1], num_filters[i], K=3, device=device, dtype=dtype)      
          if self.batchnorm:
            self.params[f'gamma{i}'] = torch.ones(num_filters[i], device=device, dtype=dtype)
            self.params[f'beta{i}'] = torch.zeros(num_filters[i], device=device, dtype=dtype)
      else:
        if i == 0:  # first conv
          self.params[f'W{i}'] = torch.normal(0.0, weight_scale, size=(num_filters[i], input_dims[0], 3, 3)).to(dtype).to(device)
          if self.batchnorm:
            self.params[f'gamma{i}'] = torch.ones(num_filters[i], device=device, dtype=dtype)
            self.params[f'beta{i}'] = torch.zeros(num_filters[i], device=device, dtype=dtype)
        elif i == self.num_layers - 1:  # linear layer
          in_dim = num_filters[-1] * ((input_dims[-1] // (2 ** len(max_pools))) ** 2)
          self.params[f'W{i}'] = torch.normal(0.0, weight_scale, size=(in_dim, num_classes)).to(dtype).to(device)
        else:  # conv
          self.params[f'W{i}'] = torch.normal(0.0, weight_scale, size=(num_filters[i], num_filters[i-1], 3, 3)).to(dtype).to(device)
          if self.batchnorm:
            self.params[f'gamma{i}'] = torch.ones(num_filters[i], device=device, dtype=dtype)
            self.params[f'beta{i}'] = torch.zeros(num_filters[i], device=device, dtype=dtype)

    for i in range(self.num_layers):
      if i == 0:  # first conv
        self.params[f'b{i}'] = torch.zeros(num_filters[i], dtype=dtype, device=device)
      elif i == self.num_layers - 1:  # linear layer
        self.params[f'b{i}'] = torch.zeros(num_classes, dtype=dtype, device=device)
      else:  # conv
        self.params[f'b{i}'] = torch.zeros(num_filters[i], dtype=dtype, device=device)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
      
    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))


  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
    scores = None

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the DeepConvNet, computing the      #
    # class scores for X and storing them in the scores variable.              #
    #                                                                          #
    # You should use the fast versions of convolution and max pooling layers,  #
    # or the convolutional sandwich layers, to simplify your implementation.   #
    ############################################################################
    # Replace "pass" statement with your code

    # {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    out = X
    outs = []
    caches = []
    w_num = self.num_layers - 1

    # conv layers
    for i in range(w_num):
      if self.batchnorm:
        if i in self.max_pools:
          out, cache = Conv_BatchNorm_ReLU_Pool.forward(out, self.params[f'W{i}'], \
            self.params[f'b{i}'], self.params[f'gamma{i}'], self.params[f'beta{i}'], conv_param, self.bn_params[i], pool_param)
        else:
          out, cache = Conv_BatchNorm_ReLU.forward(out, self.params[f'W{i}'], \
            self.params[f'b{i}'], self.params[f'gamma{i}'], self.params[f'beta{i}'], conv_param, self.bn_params[i])
        outs.append(out)
        caches.append(cache)
      else:
        if i in self.max_pools:
          out, cache = Conv_ReLU_Pool.forward(out, self.params[f'W{i}'], self.params[f'b{i}'], conv_param, pool_param)
        else:
          out, cache = Conv_ReLU.forward(out, self.params[f'W{i}'], self.params[f'b{i}'], conv_param)
        outs.append(out)
        caches.append(cache)

    # linear layer
    scores, cache = Linear.forward(out, self.params[f'W{w_num}'], self.params[f'b{w_num}'])
    outs.append(scores)
    caches.append(cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the DeepConvNet, storing the loss  #
    # and gradients in the loss and grads variables. Compute data loss using   #
    # softmax, and make sure that grads[k] holds the gradients for             #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code

    loss, da = softmax_loss(outs.pop(), y.long())

    if self.reg > 0:
      for i in range(self.num_layers):
        loss += self.reg * (torch.sum(self.params[f'W{i}'] * self.params[f'W{i}']))
        # if i < w_num:
        #   loss += self.reg * (torch.sum(self.params[f'gamma{i}'] * self.params[f'gamma{i}']))
        #   loss += self.reg * (torch.sum(self.params[f'beta{i}'] * self.params[f'beta{i}']))

    ds = {}

    ds[f'X{w_num}'], grads[f'W{w_num}'], grads[f'b{w_num}'] = Linear.backward(da, caches.pop())
    for i in reversed(range(w_num)):
      if self.batchnorm:
        if i in self.max_pools:
          ds[f'X{i}'], grads[f'W{i}'], grads[f'b{i}'], grads[f'gamma{i}'], grads[f'beta{i}'] = \
            Conv_BatchNorm_ReLU_Pool.backward(ds[f'X{i+1}'], caches.pop())
        else:
          ds[f'X{i}'], grads[f'W{i}'], grads[f'b{i}'], grads[f'gamma{i}'], grads[f'beta{i}'] = \
            Conv_BatchNorm_ReLU.backward(ds[f'X{i+1}'], caches.pop())
      else:
        if i in self.max_pools:
          ds[f'X{i}'], grads[f'W{i}'], grads[f'b{i}']  = Conv_ReLU_Pool.backward(ds[f'X{i+1}'], caches.pop())
        else:
          ds[f'X{i}'], grads[f'W{i}'], grads[f'b{i}']  = Conv_ReLU.backward(ds[f'X{i+1}'], caches.pop())

    # if self.reg > 0:
    for i in range(self.num_layers):
      grads[f'W{i}'] += 2 * self.reg * self.params[f'W{i}']
      # if i < w_num:
      #   grads[f'gamma{i}'] += 2 * self.reg * self.params[f'gamma{i}']
      #   grads[f'beta{i}'] += 2 * self.reg * self.params[f'beta{i}']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def find_overfit_parameters():
  weight_scale = 2e-3   # Experiment with this!
  learning_rate = 1e-5  # Experiment with this!
  ############################################################################
  # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
  # training accuracy within 30 epochs.                                      #
  ############################################################################
  # Replace "pass" statement with your code
  learning_rate = 1e-4
  # weight_scale = 2e-5
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
  model = None
  solver = None
  ################################################################################
  # TODO: Train the best DeepConvNet that you can on CIFAR-10 within 60 seconds. #
  ################################################################################
  # Replace "pass" statement with your code

  data_dict["X_train"] = data_dict["X_train"].to(dtype).to(device)
  data_dict["y_train"] = data_dict["y_train"].to(dtype).to(device)
  data_dict["X_val"] = data_dict["X_val"].to(dtype).to(device)
  data_dict["y_val"] = data_dict["y_val"].to(dtype).to(device)

  input_dims = data_dict['X_train'].shape[1:]

  model = DeepConvNet(input_dims=input_dims, num_classes=10,
                      num_filters=[16, 64, 64],
                      max_pools=[0, 2],
                      reg=1e-5,
                      batchnorm=False,
                      weight_scale='kaiming',
                      dtype=torch.float64,
                      device=device)

  solver = Solver(model, data_dict, batch_size=32,
                  num_epochs=30, num_train_samples=3000,
                  update_rule=adam, lr_decay=0.5,
                  optim_config={'learning_rate': 2e-3,}, device=device)

  ################################################################################
  #                              END OF YOUR CODE                                #
  ################################################################################
  return solver

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float64):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None

  gain = torch.tensor(gain).to(device)
  Din = torch.tensor(Din).to(device)

  # ref: https://www.youtube.com/watch?v=s2coXdufOzE
  # tanh --> Xavier init
  # else (esp. relu) --> He init

  if K is None:
    ###########################################################################
    # TODO: Implement Kaiming initialization for linear layer.                #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din).                                   #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code

    weight = torch.randn((Din, Dout), dtype=dtype, device=device)
    std = torch.sqrt(gain / Din)
    weight *= std

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  else:
    ###########################################################################
    # TODO: Implement Kaiming initialization for convolutional layer.         #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din) * K * K                            #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code

    weight = torch.randn((Dout, Din, K, K), dtype=dtype, device=device)
    std = torch.sqrt(gain / (Din * K * K))
    weight *= std

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  return weight

class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the PyTorch
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    # ref: https://en.wikipedia.org/wiki/Batch_normalization

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    _, D = x.shape

    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    out, cache = None, None
    if mode == 'train':
      #######################################################################
      # TODO: Implement the training-time forward pass for batch norm.      #
      # Use minibatch statistics to compute the mean and variance, use      #
      # these statistics to normalize the incoming data, and scale and      #
      # shift the normalized data using gamma and beta.                     #
      #                                                                     #
      # You should store the output in the variable out. Any intermediates  #
      # that you need for the backward pass should be stored in the cache   #
      # variable.                                                           #
      #                                                                     #
      # You should also use your computed sample mean and variance together #
      # with the momentum variable to update the running mean and running   #
      # variance, storing your result in the running_mean and running_var   #
      # variables.                                                          #
      #                                                                     #
      # Note that though you should be keeping track of the running         #
      # variance, you should normalize the data based on the standard       #
      # deviation (square root of variance) instead!                        # 
      # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
      # might prove to be helpful.                                          #
      #######################################################################
      # Replace "pass" statement with your 

      sample_mean = torch.mean(x, dim=0)
      sample_var = torch.mean((x - sample_mean) ** 2, dim=0)

      x_hat = (x - sample_mean) / torch.sqrt(sample_var + eps)
      out = gamma * x_hat + beta

      running_mean = momentum * running_mean + (1 - momentum) * sample_mean
      running_var = momentum * running_var + (1 - momentum) * sample_var

      cache = (x, x_hat, gamma, sample_mean, sample_var, eps, mode)

      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    elif mode == 'test':
      #######################################################################
      # TODO: Implement the test-time forward pass for batch normalization. #
      # Use the running mean and variance to normalize the incoming data,   #
      # then scale and shift the normalized data using gamma and beta.      #
      # Store the result in the out variable.                               #
      #######################################################################
      # Replace "pass" statement with your code

      x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
      out = gamma * x_hat + beta

      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # Don't forget to implement train and test mode separately.               #
    ###########################################################################
    # Replace "pass" statement with your code

    # ref: https://eehoeskrap.tistory.com/430

    (x, x_hat, gamma, mean, var, eps, mode) = cache
    N, _ = x.shape

    if mode == 'train':
      dbeta = torch.sum(dout, dim=0)
      dgamma = torch.sum(x_hat * dout, dim=0)
      dx_hat = dout * gamma
      dvar = torch.sum(dx_hat * (x - mean) * (-0.5), dim=0) * ((var + eps) **-1.5)
      dmean = torch.sum(dx_hat * (-1 / torch.sqrt(var + eps)), dim=0) + dvar * (torch.sum(-2 * (x - mean), dim=0)) / N
      dx = dx_hat * (1 / torch.sqrt(var + eps)) + dvar * 2 * (x - mean) / N + dmean / N
    # elif mode == 'test':
    #   dx = torch.ones((N, D))
    #   dgamma = torch.ones(D)
    #   dbeta = torch.ones(D)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # Replace "pass" statement with your code

    (x, x_hat, gamma, _, var, _, mode) = cache
    N, _ = x.shape
    if mode == 'train':
      dbeta = torch.sum(dout, dim=0)
      dgamma = torch.sum(x_hat * dout, dim=0)
      dx = (dout * gamma - torch.sum(dout * gamma, dim=0) / N - torch.sum(dout * gamma * x_hat, dim=0) * x_hat / N) / var ** (0.5)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code

    N, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C, H, W) --> (N, H, W, C)
    out, cache = BatchNorm.forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)  # (N, H, W, C) --> (N, C, H, W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code

    N, C, H, W = dout.shape
    dout = dout.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C, H, W) --> (N, H, W, C)
    dx, dgamma, dbeta = BatchNorm.backward(dout, cache)
    # dx, dgamma, dbeta = BatchNorm.backward_alt(dout, cache)  # will induce large errors but less time
    dx = dx.reshape(N, H, W, C).permute(0, 3, 1, 2)  # (N, H, W, C) --> (N, C, H, W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


################################################################################
################################################################################
#################   Fast Implementations and Sandwich Layers  ##################
################################################################################
################################################################################

class FastConv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    return dx, dw, db


class FastMaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, pool_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
    except RuntimeError:
      dx = torch.zeros_like(tx)
    return dx

class Conv_ReLU(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    out, relu_cache = ReLU.forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db


class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, conv_param, pool_param):
    """
    A convenience layer that performs a convolution, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db

class Linear_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an linear transform, batch normalization,
    and ReLU.
    Inputs:
    - x: Array of shape (N, D1); input to the linear layer
    - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
      the linear transform.
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.
    Returns:
    - out: Output from ReLU, of shape (N, D2)
    - cache: Object to give to the backward pass.
    """
    a, fc_cache = Linear.forward(x, w, b)
    a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the linear-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
    dx, dw, db = Linear.backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta
