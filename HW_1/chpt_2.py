import sys
import numpy as np


class NonStationaryBandit:
    def __init__(self, k=10):
        bandit_seed = int(np.random.randint(0, 100000000))
        self.rg = np.random.RandomState(seed=bandit_seed)
        self.k = k
        self.q_star = np.zeros(k)
    
    def best(self):  # this determines the best action to take
        # best action is always the action with maximal reward
        best_action = np.argmax(self.q_star)
        return best_action

    # Given a chosen action, give corresponding reward 
    # and take a step in the random walk (updates q* values) 
    def step(self, a):
        # sample the reward of the action (mean is q* of action, std dev is 1)
        reward = self.rg.normal(self.q_star[a], 1)

        # ALL action q* values take a step in the random walk
        self.q_star = self.q_star + self.rg.normal(0, 0.01, size=self.k) 
        return reward


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
        # greedy action selection with probability 1-epsilon (if greedy == 1) 
        greedy = np.random.binomial(1, 1 - self.epsilon)
        if greedy == 1:
            action = np.argmax(self.q)
        else:
            # random other action (equal probabilities) with probability epsilon
            action = np.random.choice(self.k)
        return action


class SampleAverage(ActionValue):
    def __init__(self, k, epsilon):
        super().__init__(k, epsilon)
        # track how many times each state has been visited
        self.visits = np.zeros(self.k) 

    def reset(self):
        self.q = np.zeros(self.k)
        self.visits = np.zeros(self.k)

    def update(self, a, r):
        # increment visit count of the chosen state, calculate new sample average
        self.visits[a] += 1
        self.q[a] = self.q[a] + ((1/self.visits[a]) * (r - self.q[a]))


class ConstantStepSize(ActionValue):
    def __init__(self, alpha, k, epsilon):
        super().__init__(k, epsilon)
        self.alpha = alpha
        
    def reset(self):
        self.q = np.zeros(self.k)

    def update(self, a, r):
        self.q[a] = self.q[a] + (self.alpha * (r - self.q[a]))


if __name__ == "__main__":
    N_bandit_runs = 300
    N_steps_for_each_bandit = 10000

    sample_average = SampleAverage(k=10, epsilon=0.1)
    constant = ConstantStepSize(k=10, epsilon=0.1, alpha=0.1)

    outputs = []

    for algo in [sample_average, constant]:
        # holds sum of rewards across all runs (one per step)
        reward_sum = [] 
        # holds % of optimal actions chosen across runs
        best_action_sum = []

        for run in range(N_bandit_runs):
            # reset all q values to 0 
            bandit = NonStationaryBandit()
            algo.reset()
            # holds the reward obtained at each step
            step_rewards = []
            # holds 1 if the optimal action was taken at the timestep, 0 if not 
            best_action_taken = [] 
            
            # each run has N_steps_for_each_bandit
            for step in range(N_steps_for_each_bandit):
                # get action from epsilon_greedy_policy
                action = algo.epsilon_greedy_policy()
                # check if action was the optimal one
                if action == bandit.best():
                    best_action_taken.append(1)
                else:
                    best_action_taken.append(0)

                # get reward from Bandit, given the action
                reward = bandit.step(a=action)
                # record the reward, to calculate the average over all runs
                step_rewards.append(reward)

                # give reward and action to algorithm, call update
                algo.update(a=action, r=reward)

            if run == 0:
                reward_sum = np.array(step_rewards) 
                best_action_sum = np.array(best_action_taken)
            else:
                reward_sum += np.array(step_rewards)
                best_action_sum += np.array(best_action_taken)

        # calculate average reward across finished runs 
        reward_avg = reward_sum / N_bandit_runs 
        # calculate % of time best action is taken over finished runs 
        best_action_avg = best_action_sum / N_bandit_runs 
        outputs += [reward_avg, best_action_avg]

    np.savetxt(sys.argv[1], outputs)