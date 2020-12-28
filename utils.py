import numpy as np
import matplotlib.pyplot as plt



# function to plot a single stroke
def plot_one_stroke(stroke,ax,label=None):
    x,y = np.cumsum(stroke[:,1]), np.cumsum(stroke[:,2])
    x_len = x.max() - x.min()
    y_len = y.max() - y.min()
    pen_lifts = np.flatnonzero(stroke[:,0])
    begin = 0
    for idx in pen_lifts:
        ax.plot(x[begin:idx],y[begin:idx], 'k')
        begin = idx + 1
    if label: ax.set_title(label,fontsize=15)
    ax.set_axis_off()

def plot_strokes(strokes,labels=None):
    if strokes.ndim == 2:
        _,ax = plt.subplots(figsize=(12,2))
        plot_one_stroke(strokes,ax,labels)
        return 
    fig,axs = plt.subplots(nrows=len(strokes),figsize=(12,2*len(strokes)))
    if not (isinstance(labels,np.ndarray) or isinstance(labels,list)): 
        labels = [None]*len(strokes)
    for i,stroke in enumerate(strokes):
        plot_one_stroke(stroke,axs[i],labels[i])
    plt.tight_layout()

def plot_lr_find(lr_list,loss_list,skip=0):
    size = len(loss_list) - skip 
    plt.plot(lr_list[:size],loss_list[:size])
    plt.xscale('log')
    plt.show()

import torch
from torch.distributions import Categorical, MultivariateNormal

def check_gpu():
    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    return device

def log_likelihood(target,rho,sigma,mu):

    const_pi = torch.as_tensor(np.pi, device=mu.device)

    diff = target.unsqueeze(-2).expand_as(mu) - mu
    Z = torch.sum(diff**2/(sigma**2), -1) - \
                    2*rho*diff.prod(-1)/(sigma.prod(-1))
    exp_terms = -Z/(2*(1-rho**2))
    log_terms = (2*const_pi).log() + (sigma[...,0]).log() + \
                    (sigma[...,1]).log() + 0.5*(1-rho**2).log()
    ans = -log_terms + exp_terms
    return  ans

# Sampling 
def ce_sample(ce,temp=1):
    ces = Categorical(logits=ce/temp).sample()
    return ces.float()

def bivariate_sample(params):
    """Sample from a mixture of bivariate gaussians"""
    pi,rho,sigma,mu = [p.squeeze(1) for p in params]
    pis = Categorical(probs=pi.exp()).sample()
    bs = torch.arange(mu.size(0))
    sample_mu = mu[bs,pis]
    covar = torch.empty((bs.size(0),2,2),device=mu.device)
    covar[bs,0,0] = sigma[bs,pis,0] ** 2
    covar[bs,0,1] = rho[bs,pis] * sigma[bs,pis,0] * sigma[bs,pis,1] 
    covar[bs,1,0] = covar[bs,0,1] 
    covar[bs,1,1] = sigma[bs,pis,1] ** 2
    sample = MultivariateNormal(sample_mu, covar).sample()
    return sample




