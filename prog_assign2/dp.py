from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

class GreedyPolicy(Policy):
    def __init__(self, env:EnvWithModel, values:np.array):
        self.env = env
        self.values = values

    # 1 for action with maximal value, 0 for every other action 
    def action_prob(self, state:int, action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        # get maximal action given the state
        _, max_action = get_maximal_action(env=self.env, state=state, values=self.values)
        if action == max_action:
            return 1.0
        else:
            return 0.0
        

    # always take the greedy action
    def action(self, state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        # get maximal action given the state
        _, action = get_maximal_action(env=self.env, state=state, values=self.values)
        return action


# given the enviornment, current state, and value estimates
# return the maximal next state expected value over all actions and the corresponding action
# if policy is provided, use action probability from policy in calculation
def get_maximal_action(env, state, values):
    trans_dynamics = env.TD
    rewards = env.R
    discount = env.spec.gamma
    # holds expected values of each action possible from given state (action is index)
    action_expected_vals = [0] * trans_dynamics[state].shape[0] 

    # indexing with current value as first dim yeilds array of dimensions: [possible actions, corresponding next states]
    for action in range(trans_dynamics[state].shape[0]):
        next_states = trans_dynamics[state][action] 
        # calculate sum of p(s', r | s,a)[r + discount * V(s')] over all possible next states s' given the action a
        expected_val = 0
        for next_state in range(len(next_states)):
            trans_prob = trans_dynamics[state, action, next_state]
            reward = rewards[state, action, next_state] 
            expected_val += trans_prob * (reward + (discount * values[next_state]))
        action_expected_vals[action] = expected_val

    return max(action_expected_vals), np.argmax(action_expected_vals)


def get_policy_eval_value(env, policy, state, values):
    # calculate expected value under policy of the state
    trans_dynamics = env.TD
    rewards = env.R
    discount = env.spec.gamma
    state_expected_value = 0

    # indexing with current value as first dim yeilds array of dimensions: [possible actions, corresponding next states]
    for action in range(trans_dynamics[state].shape[0]):
        next_states = trans_dynamics[state][action] 
        action_prob = policy.action_prob(state=state, action=action)
        action_expected_val = 0
        for next_state in range(len(next_states)):
            trans_prob = trans_dynamics[state, action, next_state]
            reward = rewards[state, action, next_state] 
            action_expected_val += trans_prob * (reward + (discount * values[next_state])) 
        state_expected_value += action_prob * action_expected_val 
    
    return state_expected_value
        

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    values = initV
    # done when this is less than theta
    delta = float('inf')

    while delta > theta:
        delta = 0
        # each index holds a state value
        for i in range(len(values)):
            value = values[i]
            # find the state estimate (maximal action's Bellman update with the given policy)
            v_estimate = get_policy_eval_value(env=env, policy=pi, state=i, values=values)
            values[i] = v_estimate
            # delta can only get bigger for each state 
            delta = max(delta, abs(value - v_estimate))

    # create Q function based on the policy's estimated values
    Q = np.zeros([env.spec.nS, env.spec.nA])
    trans_probs = env.TD
    rewards = env.R
    discount = env.spec.gamma
    for state in range(Q.shape[0]):
        actions = Q[state]
        for action in range(len(actions)):
            next_states = trans_probs[state][action]
            # calculate expected value of the action state pair (transition dynamic * (reward + discounted value function))
            # value function is from policy, so Q function corresponds to this policy
            expected_val = 0
            for next_state in range(len(next_states)):
                trans_prob = trans_probs[state, action, next_state]
                reward = rewards[state, action, next_state]
                expected_val += trans_prob * (reward + (discount * values[next_state])) 
            Q[state, action] = expected_val

    return values, Q


def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """
    values = initV
    # done when this is less than theta
    delta = float('inf')

    while delta > theta:
        delta = 0
        # each index holds a state value
        for i in range(len(values)):
            value = values[i]
            # find the state estimate (maximal action's Bellman update)
            v_estimate, _ = get_maximal_action(env=env, state=i, values=values) 
            values[i] = v_estimate
            # delta can only get bigger for each state 
            delta = max(delta, abs(value - v_estimate))

    # once values stop changing, create greedy policy with the values
    pi = GreedyPolicy(env=env, values=values)

    return values, pi
