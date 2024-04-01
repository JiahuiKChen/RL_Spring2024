import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_high = state_high
        self.tile_width = tile_width
        self.state_low = state_low
        self.num_tilings = num_tilings
        self.num_actions = num_actions

        self.tiles_per_dim = (np.ceil(((self.state_high - self.state_low) / self.tile_width) + 1)).astype(int)
        self.num_tiles = np.prod(self.tiles_per_dim)
        self.tiling_offset = self.tile_width / self.num_tilings
        self.offsets = [self.state_low - i * self.tiling_offset for i in range(self.num_tilings)]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        len = self.num_actions * self.num_tilings * self.num_tiles 
        return len

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        x = np.zeros(self.feature_vector_len())
        # return all 0s for terminal state
        if done:
            return x
        
        len_per_action = self.num_tiles * self.num_tilings
        x_action_tiles = x[a * len_per_action : (a + 1) * len_per_action]

        for offset_ind in range(len(self.offsets)):
            offset = self.offsets[offset_ind]
            x_action_tiling = x_action_tiles[offset_ind * self.num_tiles : (offset_ind+1) * self.num_tiles]
            multi_ind = tuple( ((s - offset) // self.tile_width ).astype(int))
            flat_ind = np.ravel_multi_index(multi_ind, self.tiles_per_dim)

            x_action_tiling[flat_ind] = 1

        return x


def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for _ in range(num_episode):
        init_state, _ = env.reset()
        action = epsilon_greedy_policy(init_state, False, w)
        x = X(init_state, False, action)

        # initialize trace vector (z) and dot product between feature vector and weights (q)
        z = 0
        q_old = 0
        terminal = False

        while not terminal:
            next_state, reward, terminal, _ = env.step(action)
            next_action = epsilon_greedy_policy(next_state, terminal, w)
            next_x = X(next_state, terminal, next_action)

            q = np.dot(w, x)
            q_next = np.dot(w, next_x)
            delta = reward + (gamma * (q_next - q))

            # update weights, trace vector
            z = (gamma * lam * z) + (1 - (alpha * gamma * lam * np.dot(z, x))) * x
            w = w + alpha*(delta + (q - q_old))*z - alpha*(q - q_old)*x

            q_old = q_next
            x = next_x
            action = next_action
        
    return w
