from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

class QPolicy(Policy):
    def __init__(self, q:np.array):
        self.q = q
    
    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        possible_actions = self.q[state]
        greedy_action = np.argmax(possible_actions)
        if action == greedy_action:
            return 1.0
        else:
            return 0.0

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        possible_actions = self.q[state]
        action = np.argmax(possible_actions)
        return action

    def update_q(self, q:np.array):
        self.q = q


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
        # tracks reward received at each time step (needed since we look backwards to calculate goal)
        rewards = [0] 
        # tracks state at each time step (needed since we look backwards to update states) 
        states = []
        states.append(episode_traj[0][0])
        step = 0

        while True:
            if step < terminal:
                # store reward and state at each step 
                rewards.append(episode_traj[step][2])
                states.append(episode_traj[step][3])

                # if next state is terminal, then we're at the end of the trajectory
                if step == (len(episode_traj) -1): 
                    terminal = step + 1

            update_time = step - n + 1

            # if we've already seen n steps, we can start updating values
            if update_time >= 0:
                update_state = int(states[update_time])

                goal = 0
                for i in range(update_time + 1, min(update_time + n, terminal) + 1):
                    goal += (discount**(i - update_time - 1)) * rewards[i] 
                
                if (update_time + n) < terminal:
                    last_state = states[update_time + n]
                    goal += (discount**n) * values[last_state]

                values[update_state] = values[update_state] + (alpha * (goal - values[update_state]))
            
            if update_time == terminal - 1:
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
    q = initQ
    discount = env_spec.gamma
    pi = QPolicy(q=q)

    # each episode trajectory is a list of tuples: (state, action, reward, next state) 
    for episode_traj in trajs:
        terminal = float('inf')
        # tracks reward received at each time step (needed since we look backwards to calculate goal)
        rewards = [0] 
        # tracks state at each time step (needed since we look backwards to update states) 
        states = []
        actions = []
        states.append(episode_traj[0][0])
        actions.append(episode_traj[0][1])
        step = 0

        while True:
            if step < terminal:
                # store reward and state at each step 
                rewards.append(episode_traj[step][2])
                states.append(episode_traj[step][3])

                # if next state is terminal, then we're at the end of the trajectory
                if step == (len(episode_traj) -1): 
                    terminal = step + 1
                else:
                    actions.append(episode_traj[step+1][1])

            update_time = step - n + 1

            # if we've already seen n steps, we can start updating values
            if update_time >= 0:
                goal = 0
                for i in range(update_time + 1, min(update_time + n, terminal) + 1):
                    goal += (discount**(i - update_time - 1)) * rewards[i] 

                rho = 1
                for i in range(update_time + 1, min(update_time + n, terminal-1) + 1):
                    action_i = actions[i]
                    state_i = states[i]
                    rho = rho * (pi.action_prob(state=state_i, action=action_i)
                                 /
                                 bpi.action_prob(state=state_i, action=action_i)) 

                if (update_time + n) < terminal:
                    last_state = states[update_time + n]
                    last_action = actions[update_time + n]
                    goal += (discount**n) * q[last_state, last_action]

                update_state = states[update_time]
                update_action = actions[update_time]
                q[update_state, update_action] = q[update_state, update_action] + \
                    (alpha * rho * (goal - q[update_state, update_action]))
                # once Q is updated, update policy
                pi.update_q(q=q)
            
            if update_time == terminal - 1:
                break
            else:
                step += 1

    return q, pi
