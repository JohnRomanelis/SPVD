import torch
import random 
import numpy as np


class NoiseSchedulerDDPM:

    def __init__(self, beta_min, beta_max, n_steps, mode='linear'):
        self.n_steps, self.beta_min, self.beta_max = n_steps, beta_min, beta_max
        
        if mode == 'linear':
            self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
            self.α = 1. - self.beta 
            self.ᾱ = torch.cumprod(self.α, dim=0)
            #self.σ = self.beta.sqrt()
        elif mode == "warm0.1":
            self.beta = beta_max * torch.ones(n_steps, dtype=torch.float)
            warmup_time = int(0.1 * n_steps)
            self.beta[:warmup_time] = torch.linspace(beta_min, beta_max, warmup_time, dtype=torch.float)
            self.α = 1. - self.beta
            self.ᾱ = torch.cumprod(self.α, dim=0)
            #self.σ = self.beta.sqrt()
        
        self.beta = self.beta.numpy()
        self.α = self.α.numpy()
        self.ᾱ = self.ᾱ.numpy() 
        #self.σ = self.σ.numpy()

    def __call__(self, x0):

        # select random timestep
        t = random.randint(0, self.n_steps-1)

        # random noise
        ε = torch.randn(x0.shape)

        # interpolate noise
        ᾱ_t = self.ᾱ[t]
                
        xt = np.sqrt(ᾱ_t)*x0 + np.sqrt(1-ᾱ_t)*ε

        return xt, t, ε