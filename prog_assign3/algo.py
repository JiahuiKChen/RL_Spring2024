import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + alpha[G- \hat{v}(s_tau;w)] nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: episodes to iterate
    output:
        None
    """
    discount = gamma

    for episode in range(num_episode):
        init_state, _ = env.reset()
        # tracks state at each time step (needed since we look backwards to update states) 
        states = [np.array([init_state, 0.0])]
        # tracks reward received at each time step (needed since we look backwards to calculate goal)
        rewards = [0] 
        terminal = float('inf')
        step = 0

        while True:
            if step < terminal:
                # get action from policy, given current state
                action = pi.action(state=states[step]) 
                next_state, reward, terminated, _ = env.step(action)
                # store reward and state at each step 
                states.append(next_state)
                rewards.append(reward)

                # if next state is terminal, then we're at the end of the trajectory
                if terminated: 
                    terminal = step + 1

            update_time = step - n + 1

            # if we've already seen n steps, we can start updating values
            if update_time >= 0:
                update_state = states[update_time]

                goal = 0
                for i in range(update_time + 1, min(update_time + n, terminal) + 1):
                    goal += (discount**(i - update_time - 1)) * rewards[i] 
                
                if (update_time + n) < terminal:
                    last_state = states[update_time + n]
                    goal += (discount**n) * V(last_state)

                V.update(alpha=alpha, G=goal, s_tau=update_state)
            
            if update_time == terminal - 1:
                break
            else:
                step += 1
