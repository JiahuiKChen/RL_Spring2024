from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

class QPolicy(Policy):
    def __init__(self, q:np.array):
        #####################
        # TODO: Implement the methods in this class.
        # You may add any arguments to the constructor as you see fit.
        # "QPolicy" here refers to a policy that takes 
        #    greedy actions w.r.t. Q values
        #####################
        raise NotImplementedError()
    
    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        raise NotImplementedError()

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError()

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """
    values = initV
    discount = env_spec.gamma

    # each episode trajectory is a list of tuples: (state, action, reward, next state) 
    for episode_traj in trajs:
        terminal = float('inf')
        rewards = np.zeros(len(episode_traj))
        update_state = 0
        step = 0

        while True:
            if step < terminal:
                # store reward at each step so we can use it to calculate goal when n steps ahead
                rewards[step % (n+1)] = episode_traj[step][2]

                # if next state is terminal, then we're at the end of the trajectory
                if step == (len(episode_traj) -1): 
                    terminal = step + 1

            update_state = step - n + 1

            # if we've already seen n steps, we can start updating values
            if update_state >= 0:
                goal = 0
                for i in range(update_state + 1, min(update_state + n, terminal)):
                    goal += (discount**(i - update_state - 1)) * rewards[i % (n+1)] 
                
                if (update_state + n) < terminal:
                    goal += (discount**n) * values[(update_state + n) % (n + 1)]

                values[update_state % (n+1)] = values[update_state % (n+1)] + (alpha * (goal - values[update_state % (n+1)]))
            
            if update_state == terminal - 1:
                break
            else:
                step += 1

    return values

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    return Q, pi
