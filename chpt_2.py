import sys
import numpy as np


class NonStationaryBandit:
    def __init__(self, k=10, bandit_seed=0):
        self.rg = np.random.RandomState(seed=bandit_seed)
        self.q_star = np.zeros(k)
    
    def reset(self, episode_seed=None): # based on seed, different episode will be generated
        if episode_seed is None:
            episode_seed = int(self.rg.randint(0, 100000000))

        self.episode_rg = np.random.RandomState(seed=episode_seed)
        
    def best(self):  # this determines the best action to take
        # TODO: your code goes here
        raise NotImplementedError

    def step(self, a):
        # TODO: your code goes here
        # hint: you can use self.episode_rg to generate noise
        raise NotImplementedError


class ActionValue:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k)

    def reset(self):
        # no code needed, but implement this method in the child classes
        raise NotImplementedError

    def update(self, a, r):
        # no code needed, but implement this method in the child classes
        raise NotImplementedError

    def epsilon_greedy_policy(self):
        # TODO: your code goes here
        raise NotImplementedError


class SampleAverage(ActionValue):
    def __init__(self, k, epsilon):
        super().__init__(k, epsilon)
        # TODO: your code goes here
        raise NotImplementedError

    def reset(self):
        # TODO: your code goes here
        raise NotImplementedError

    def update(self, a, r):
        # TODO: your code goes here
        raise NotImplementedError


class ConstantStepSize(ActionValue):
    def __init__(self, alpha, k, epsilon):
        super().__init__(k, epsilon)
        # TODO: your code goes here
        raise NotImplementedError
        
    def reset(self):
        # TODO: your code goes here
        raise NotImplementedError

    def update(self, a, r):
        # TODO: your code goes here
        raise NotImplementedError


def experiment(bandit, algorithm, steps, episode_seed=None):

    bandit.reset(episode_seed)
    algorithm.reset()

    rs = []
    best_action_taken = []

    # TODO: implement the experiment loop

    return np.array(rs), np.array(best_action_taken)

if __name__ == "__main__":
    N_bandit_runs = 300
    N_steps_for_each_bandit = 10000

    sample_average = SampleAverage(k=10, epsilon=0.1)
    constant = ConstantStepSize(k=10, epsilon=0.1, alpha=0.1)

    outputs = []

    for algo in [sample_average, constant]:
        # TODO: run multiple experiments (where N = N_bandit_runs)
        # you will need to compute the average reward across all experiments
        # you will also compute the percentage of times the best action is taken

        raise NotImplementedError

        outputs += [average_rs, average_best_action_taken]

    np.savetxt(sys.argv[1], outputs)