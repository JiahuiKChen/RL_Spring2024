from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import numpy as np

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()

        # 2 hidden layers with 32 nodes, ReLU as activations for hidden layers
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Linear(32, num_actions),
            # softmax over number of actions is output
            nn.Softmax(dim=-1)
        ) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=alpha)

    def forward(self, states, return_prob=False):
        # Note: You will want to return either probabilities or an action
        # Depending on the return_prob parameter
        # This is to make this function compatible with both the
        # update function below (which needs probabilities)
        # and because in test cases we will call pi(state) and 
        # expect an action as output.
        # self.model.eval()

        if isinstance(states, np.ndarray):
            input = torch.Tensor(states)
        else:
            input = states
        action_probs = self.model(input)

        if not return_prob:
            action_probs = action_probs.detach()
            action = torch.argmax(action_probs, dim=-1).numpy()
            return action
        else:
            return action_probs

    def update(self, states, actions_taken, gamma_t, delta):
        """
        states: states
        actions_taken: actions_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.model.train()
        action_prob = self.forward(states, return_prob=True)
        policy_loss = torch.mean(-torch.log(action_prob) * delta * gamma_t)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    There is no need to change this class.
    """
    def __init__(self,b):
        self.b = b
        
    def __call__(self, states):
        return self.forward(states)
        
    def forward(self, states) -> float:
        return self.b

    def update(self, states, G):
        pass


class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        # 2 hidden layers with 32 nodes, ReLU as activations for hidden layers
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Linear(32, 1)
        )
        # self.model.double()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=alpha)

    def forward(self, states) -> float:
        self.model.eval()
        if isinstance(states, np.ndarray):
            input = torch.Tensor(states)
        else:
            input = states
        pred = self.model(input)
        return pred.detach()

    def update(self, states, G):
        self.model.train()
        if isinstance(states, np.ndarray):
            input = torch.Tensor(states)
        else:
            input = states
        pred = self.model(input)
        if type(G) is not torch.Tensor:
            G = torch.Tensor([G])
            # if type(G) is float:
            #     G = torch.Tensor([G]) #, dtype=torch.double)
            # else:
            #     G = torch.Tensor(G) 
        loss = self.loss_func(pred, G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    # without baseline this is just the Baseline dummy class that returns 0 
    V:VApproximationWithNN) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # goal for state 0 for each epiosde (what's returned)
    G_0 = []

    for episode in range(num_episodes):
        # generate all steps for the episode
        init_state = env.reset()
        states = [init_state]
        actions = []
        rewards = [0]
        terminal = False

        while not terminal:
            action = pi(states[-1])
            state, reward, terminal, _ = env.step(action)

            if not terminal:
                states.append(state)
            actions.append(action)
            rewards.append(reward)

        # train the networks (value and policy) using the episode trajectory
        steps = len(states)
        for t in range(steps):
            G = np.sum([gamma**(k-t-1) * rewards[k] for k in range(t+1, steps+1)])
            delta = G - V(states[t])

            V.update(states=states[t], G=G)
            pi.update(states=states[t], actions_taken=actions[t], gamma_t=gamma**t, delta=delta)

            if t == 0:
                G_0.append(G)
    
    return G_0