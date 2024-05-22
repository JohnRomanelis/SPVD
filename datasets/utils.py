import torch
import random 
import numpy as np


class NoiseSchedulerDDPM:

    def __init__(self, beta_min, beta_max, n_steps):
        self.n_steps, self.beta_min, self.beta_max = n_steps, beta_min, beta_max
        
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.α = 1. - self.beta 
        self.ᾱ = torch.cumprod(self.α, dim=0)
        self.σ = self.beta.sqrt()
        
        self.beta = self.beta.numpy()
        self.α = self.α.numpy()
        self.ᾱ = self.ᾱ.numpy() 
        self.σ = self.σ.numpy()

    def __call__(self, x0):

        # select random timestep
        t = random.randint(0, self.n_steps-1)

        # random noise
        ε = torch.randn(x0.shape)

        # interpolate noise
        ᾱ_t = self.ᾱ[t]
                
        xt = np.sqrt(ᾱ_t)*x0 + np.sqrt(1-ᾱ_t)*ε

        return xt, t, ε