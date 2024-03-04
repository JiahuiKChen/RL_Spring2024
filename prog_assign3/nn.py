import numpy as np
import torch.nn as nn
import torch
from algo import ValueFunctionWithApproximation

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self, state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # 2 hidden layers with 32 nodes, ReLU as activations for hidden layers
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Linear(32, 1)
        )
        self.model.double()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=0.001)

    def __call__(self,s):
        self.model.eval()
        input = torch.from_numpy(s)
        pred = self.model(input)
        return pred.detach()

    def update(self, alpha, G, s_tau):
        self.model.train()
        input = torch.from_numpy(s_tau)
        pred = self.model(input)
        if type(G) is not torch.Tensor:
            if type(G) is float:
                G = torch.tensor([G], dtype=torch.double)
            else:
                G = torch.from_numpy(G) 
        loss = self.loss_func(pred, G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None



