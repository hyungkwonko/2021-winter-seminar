from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def hello_vae():
    print("Hello from vae.py!")


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.hidden_dim1 = 512
        self.hidden_dim2 = 256
        self.hidden_dim3 = 64
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.BatchNorm1d(self.hidden_dim3),
            nn.ReLU()
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(self.hidden_dim3, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )

        self.logvar_layer = nn.Sequential(
            nn.Linear(self.hidden_dim3, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )
        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_dim3),
            nn.BatchNorm1d(self.hidden_dim3),
            nn.ReLU(),
            nn.Linear(self.hidden_dim3, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(-1, (1, 28, 28))
        )
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.mu_layer.apply(init_weights)
        self.logvar_layer.apply(init_weights)

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code

        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar

    def sample(self, z):
        gen_imgs = self.decoder(z)
        return gen_imgs


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C

        self.hidden_dim1 = 512
        self.hidden_dim2 = 256
        self.hidden_dim3 = 64

        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms the image--after  #
        # flattening and now adding our one-hot class vector (N, H*W + C)--into a hidden_dimension #               #
        # (N, H_d) feature space, and a final two layers that project that feature space           #
        # to posterior mu and posterior log-variance estimates of the latent space (N, Z)          #
        ############################################################################################
        # Replace "pass" statement with your code
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size + self.num_classes, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.BatchNorm1d(self.hidden_dim3),
            nn.ReLU()
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(self.hidden_dim3, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )

        self.logvar_layer = nn.Sequential(
            nn.Linear(self.hidden_dim3, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )

        ############################################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that transforms the  #
        # latent space (N, Z + C) to the estimated images of shape (N, 1, H, W).                   #
        ############################################################################################
        # Replace "pass" statement with your code
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.num_classes, self.hidden_dim3),
            nn.BatchNorm1d(self.hidden_dim3),
            nn.ReLU(),
            nn.Linear(self.hidden_dim3, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(-1, (1, 28, 28))
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.mu_layer.apply(init_weights)
        self.logvar_layer.apply(init_weights)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the concatenation of input batch and one hot vectors through the encoder model  #
        # to get posterior mu and logvariance                                                      #
        # (2) Reparametrize to compute the latent vector z                                         #
        # (3) Pass concatenation of z and one hot vectors through the decoder to resconstruct x    #
        ############################################################################################
        # Replace "pass" statement with your code

        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, c], dim=1)
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = reparametrize(mu, logvar)
        z = torch.cat([z, c], dim=1)
        x_hat = self.decoder(z)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar



def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code

    epsilon = torch.randn(mu.shape, device=mu.device)
    z = torch.exp(logvar / 2) * epsilon + mu  # works for Conditional VAE
    # z = logvar * epsilon + mu  # works for Vanilla VAE

    print(z.mean(dim=0))

    # Vanilla VAE에 std쓰면 안되는 이유
    # z = logvar * epsilon + mu 이렇게 하면 되고
    # z = std * epsilon + mu 이렇게 하면 안될까?
    # mu를 출력을 해보면, latent=5 라고 하면 [0.1188, 0.6053, 0.0000, 0.1911, 0.2571] 이런 식이다.
    # std는 출력을 해보면, latent=5 라고 하면 [1.6428, 1.9556, 1.0000, 1.1322, 1.0000] 이런 식이다.
    # logvar는 출력을 해보면, latent=5 라고 하면 [0.0000, 0.0000, 0.9187, 0.3889, 0.1121] 이런 식이다.
    # 왜 std가 무조건 1이상으로 크게 나오냐면 exp를 씌워서 그럼
    # 결론: mu에 비해서 std가 너무 큰데, 때문에 sample(inference)시의 z 값과 reparam으로 나오는 z에 차이가 있음.
    # 그렇다고 loss에서 튀어버리는것은 더 큰 문제가 되기 때문에 log(std^2)에서 std = 0이면 -inf되니까
    # 최종 결론: 어쩔 수 없으니 그냥 logvar *epsilon으로 하길 추천

    # Conditional VAE에 logvar쓰면 안되는 이유
    # 이건 잘 모르겠음... 정석대로면 std 쓰는게 사실 맞는 거니까 그래서 그런 것 같음

    # -----------
    # Vanilla VAE / Conditional VAE 둘 다 z 값 거의 똑같음
    # mu의 mean: 0.3932 --> training이 진행됨에 따라 떨어져서 0.02~0.06 적당히 수렴
    # logvar의 mean: 0.3932 --> training이 진행됨에 따라 떨어져서 0.02~0.06 적당히 수렴
    # std의 mean: 1.2838 --> training이 진행됨에 따라 떨어져서 1.0486 적당히 수렴



    # Vanilla VAE, LATENT = 15, 각 mean값들

    # MU --> 0에 근점
    # [0.0061, 0.0086, 0.0145, 0.0318, 0.0018, 0.0043, 0.0126, 0.0103, 0.0017, 0.0020, 0.0118, 0.0137, 0.0133, 0.0005, 0.0020]
    # [0.0113, 0.0139, 0.0199, 0.0098, 0.0070, 0.0114, 0.0213, 0.0168, 0.0038, 0.0071, 0.0233, 0.0221, 0.0191, 0.0040, 0.0042]
    # [0.0045, 0.0098, 0.0158, 0.0394, 0.0048, 0.0054, 0.0145, 0.0122, 0.0011, 0.0031, 0.0151, 0.0156, 0.0133, 0.0012, 0.0010]

    # LOGVAR --> 0에 근점
    # [0.0047, 0.0040, 0.0090, 0.0109, 0.0030, 0.0027, 0.0018, 0.0038, 0.0021, 0.0012, 0.0032, 0.0023, 0.0019, 0.0097, 0.0034]
    # [0.0015, 0.0043, 0.0040, 0.0046, 0.0000, 0.0017, 0.0003, 0.0016, 0.0011, 0.0010, 0.0022, 0.0015, 0.0025, 0.0037, 0.0026]
    # [0.0038, 0.0082, 0.0284, 0.0308, 0.0090, 0.0182, 0.0202, 0.0207, 0.0176, 0.0156, 0.0173, 0.0177, 0.0152, 0.0297, 0.0187]

    # STD (= exp(logvar / 2)) --> 무조건 1 이상
    # [1.0000, 1.0000, 1.0001, 1.0001, 1.0050, 1.0000, 1.0000, 1.0000, 1.0000, 1.0025, 1.0000, 1.0000, 1.0001, 1.0000, 1.0000]
    # [1.0001, 1.0013, 1.0003, 1.0002, 1.0121, 1.0000, 1.0002, 1.0002, 1.0004, 1.0149, 1.0003, 1.0004, 1.0001, 1.0004, 1.0001]
    # [1.0019, 1.0054, 1.0020, 1.0029, 1.0053, 1.0025, 1.0026, 1.0019, 1.0020, 1.0063, 1.0039, 1.0028, 1.0030, 1.0026, 1.0014]

    # Z (MU + eps * LOGVAR) --> 문제 없음
    # [0.0489, 0.0464, 0.0232, 0.0366, 0.0165, 0.0281, 0.0373, 0.0513, 0.0402, 0.0371, 0.0374, 0.0319, 0.0535, 0.0311, 0.0390]
    # [0.0088, 0.0040, 0.0025, 0.0032, 0.0020, 0.0027, 0.0003, 0.0089, 0.0045, 0.0026, 0.0018, 0.0010, 0.0108, 0.0007, 0.0008]
    # [0.0138, 0.0090, 0.0008, 0.0072, -0.0002, 0.0091, 0.0022, 0.0157, 0.0086, 0.0068, 0.0052, 0.0031, 0.0208, 0.0039, 0.0038]

    # Z (MU + eps * STD) —> 결과가 확실히 튀는 것을 볼 수 있음.
    # [-0.0769,  0.0506, -0.0209,  0.2089, -0.0321,  0.1222, -0.0046, -0.0518, -0.0610, -0.0683,  0.0157,  0.2169, -0.0074,  0.0523, -0.0193]
    # [-0.0252, -0.0179,  0.0304,  0.1565, -0.1242, -0.1708, -0.1082,  0.0412, -0.0047,  0.0340,  0.1123,  0.0613,  0.0780, -0.0970, -0.1272]
    # [ 0.0211,  0.0239, -0.1606,  0.0393, -0.0748,  0.0284,  0.0495, -0.1222, -0.0195,  0.1468,  0.1009,  0.1516, -0.0292, -0.0604, -0.1375]



    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    loss = None
    ################################################################################################
    # TODO: Compute negative variational lowerbound loss as described in the notebook              #
    ################################################################################################
    # Replace "pass" statement with your code

    recon_loss = nn.functional.mse_loss(x_hat, x)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    # kld_loss = torch.mean(-0.5 * torch.sum(1 + 2 * torch.log(logvar + 1e-2) - mu ** 2 - logvar ** 2, dim=1), dim=0)  # unstable
    loss = recon_loss + kld_loss

    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss

