import numpy as np
import matplotlib.pyplot as plt



# function to plot a single stroke
def plot_stroke(stroke, label=None):
    x,y = np.cumsum(stroke[:,1]), np.cumsum(stroke[:,2])
    x_len = x.max() - x.min()
    y_len = y.max() - y.min()
    plt.figure(figsize=(2*x_len/y_len,3))
    pen_lifts = np.flatnonzero(stroke[:,0])
    begin = 0
    for idx in pen_lifts:
        plt.plot(x[begin:idx],y[begin:idx], 'k')
        begin = idx + 1
    if label is not None:
        plt.title(label,fontsize=15) 
    plt.tight_layout()
    plt.axis('off')
    plt.show()

import torch
from torch.distributions import Categorical, MultivariateNormal
# Sampling 

def ce_sample(ce,temp=1):
    ces = Categorical(logits=ce/temp).sample().data
    return ces.float()

def bivariate_sample(pi,rho,sigma,mu):
    """Sample from a mixture of bivariate gaussians"""
    pi,rho,sigma,mu = pi.squeeze(),rho.squeeze(),sigma.squeeze(),mu.squeeze()
    pis = Categorical(probs=pi.exp()).sample().data
    bs = torch.arange(sigma.size(0))
    sample_mu = mu[bs,pis]
    covar = torch.empty((sigma.size(0),2,2), device=sigma.device)
    covar[bs,0,0] = sigma[bs,pis,0] ** 2
    covar[bs,0,1] = rho[bs,pis] * sigma[bs,pis,0] * sigma[bs,pis,1] 
    covar[bs,1,0] = rho[bs,pis] * sigma[bs,pis,0] * sigma[bs,pis,1] 
    covar[bs,1,1] = sigma[bs,pis,1] ** 2
    sample = MultivariateNormal(sample_mu, covar).sample()
    return sample


def get_probs(target,rho,sigma,mu):
    # better stability
    eps = 1e-6
    rho = rho/(1+eps)
    sigma = sigma + eps
    const_pi = torch.as_tensor(np.pi, device=mu.device)

    diff = target.unsqueeze(-2).expand_as(mu) - mu
    Z = torch.sum(diff**2/(sigma**2), -1) - \
                    2*rho*diff.prod(-1)/(sigma.prod(-1))
    exp_terms = -Z/(2*(1-rho**2))
    log_terms = (2*const_pi).log() + (sigma[...,0]).log() + \
                    (sigma[...,1]).log() + 0.5*(1-rho**2).log()
    ans = -log_terms + exp_terms
    return  ans