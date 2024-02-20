from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """
    q = initQ
    # C in book pg 110 algorithm
    discount = env_spec.gamma
    # number of times each state-action pair has been visited (total steps so across episodes) 
    visits = np.zeros(q.shape)
    
    # each episode trajectory is a list of tuples: (state, action, reward, next state) 
    for episode_traj in trajs:
        goal = 0
        importance_ratio = 1
        # go through each step in the epiosde backwards, as we need the goal
        for step in reversed(range(len(episode_traj))):
            tup = episode_traj[step]
            state = tup[0]
            action = tup[1]
            reward = tup[2]
            
            visits[state, action] += 1
            goal = (discount * goal) + reward
            q[state, action] = q[state, action] + ((importance_ratio / visits[state, action]) * (goal - q[state, action]))
            importance_ratio = importance_ratio * (pi.action_prob(state=state, action=action) / bpi.action_prob(state=state, action=action))
            # if weight (aka importance sampling ratio) is ever 0, it will always be 0 and q won't change
            if importance_ratio == 0:
                continue

    return q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
            (list of N lists of tuples)
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """
    q = initQ
    # C in book pg 110 algorithm
    weight_sum = np.zeros(q.shape)
    discount = env_spec.gamma

    # each episode trajectory is a list of tuples: (state, action, reward, next state) 
    for episode_traj in trajs:
        goal = 0
        weight = 1
        # go through each step in the epiosde backwards, as we need the goal
        for step in reversed(range(len(episode_traj))):
            tup = episode_traj[step]
            state = tup[0]
            action = tup[1]
            reward = tup[2]

            goal = (discount * goal) + reward
            weight_sum[state, action] = weight_sum[state, action] + weight
            q[state, action] = q[state, action] + ((weight / weight_sum[state, action]) * (goal - q[state, action]))
            weight = weight * (pi.action_prob(state=state, action=action) / bpi.action_prob(state=state, action=action))
            # if weight (aka importance sampling ratio) is ever 0, it will always be 0 and q won't change
            if weight == 0:
                continue

    return q