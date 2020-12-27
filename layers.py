import torch
import torch.nn as nn


def trunc_normal_(x, mean:float=0., std:float=1.):
    "Truncated normal initialization."
    return x.normal_().fmod_(2).mul_(std).add_(mean)

def embedding(ni,nf,padding_idx=None):
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf, padding_idx)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb

class InLayer(nn.Module):
    def __init__(self,embed_dims=5,fc_size=10):
        super().__init__()
        # 0,1,pad
        self.embed = embedding(3,embed_dims, padding_idx=2)
        self.fc1 = nn.Linear(2,embed_dims)
        self.fc2 = nn.Linear(2*embed_dims,fc_size)
        self.activation = nn.ReLU()
    def forward(self,x):
        ordinal, real = x[...,0].long(), x[...,1:]
        ordinal = self.embed(ordinal)
        real = self.fc1(real)
        final = self.fc2(torch.cat([ordinal,real],2))
        return self.activation(final) 


class RecurLayer(nn.Module):
    def __init__(self,ip_dims=10,num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.rnns = nn.ModuleList([nn.LSTM(ip_dims,ip_dims,batch_first=True)])
        for _ in range (num_layers-1):
            self.rnns.append(nn.LSTM(ip_dims*2,ip_dims,batch_first=True))
    def forward(self,x,prev_state=None):
        if prev_state == None:
            prev_state = [None]*self.num_layers
        cat_xs, new_state = [], []
        skip_x = x.clone()
        for i in range(self.num_layers):
            new_x,s = self.rnns[i](x,prev_state[i])
            cat_xs.append(new_x), new_state.append(s)
            x = torch.cat([skip_x,new_x],2) # skip connection inputs
        return torch.cat(cat_xs, 2), new_state # outputs from all layers


class MixtureDensity(nn.Module):
    def __init__(self,ip_dims,num):
        super().__init__()
        self.num = num
        self.ce = nn.Linear(ip_dims,3)
        self.pi = nn.Linear(ip_dims,num)
        self.sigma = nn.Linear(ip_dims,num*2)
        self.rho = nn.Linear(ip_dims,num)
        self.mu = nn.Linear(ip_dims,num*2)
    def forward(self,x):
        ce = self.ce(x)
        pi = self.pi(x)
        rho = self.rho(x)
        shape = x.size()
        sigma = self.sigma(x).view(shape[0],shape[1],self.num,2)
        mu = self.mu(x).view(shape[0],shape[1],self.num,2)
        return ce,pi,rho,sigma,mu


class FinalActivation(nn.Module):
    def __init__(self,beta):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.softplus = nn.Softplus(beta=beta)
        self.tanh = nn.Tanh()
    def forward(self,ce,pi,rho,sigma,mu,bias=0,eps=1e-6):
        # eps for numerical stability req. for calc. likelihood
        pi = self.softmax(pi*(1+bias))
        rho = self.tanh(rho)/(1+eps)
        sigma = self.softplus(sigma-bias) + eps
        return ce,pi,rho,sigma,mu