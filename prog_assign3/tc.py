import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_high = state_high
        self.tile_width = tile_width
        self.state_low = state_low
        # number of tiles per dim, as each tiling covers space entirely
        self.tiles_per_dim = np.ceil((self.state_high - self.state_low) / self.tile_width).astype(int) + 1
        self.w = np.zeros((num_tilings, np.prod(self.tiles_per_dim)))
        self.num_tilings = num_tilings
       

    def __call__(self,s):
        # inner product of feature vector for state s and w 
        # since this is a linear method (pg. 205)
        x = self.get_feature_vec(s=s)
        value_approx = np.inner(self.w.flatten(), x.flatten()) 
        return value_approx 

    # construct tile coding feature vector, given the state
    def get_feature_vec(self, s):
        feature_vec = np.zeros((self.num_tilings, np.prod(self.tiles_per_dim)))

        # for each tiling, find which tile corresponds to the given state and set it to 1
        for tiling_ind in range(self.num_tilings):
            # find the low/starting value of the tiling 
            tiling_low = (self.state_low - (tiling_ind / self.num_tilings)) / (self.tiles_per_dim)
            # convert the state values into tile indices
            # s_ind = np.ceil( ((s - self.state_low) / self.tiles_per_dim) + tiling_low).astype(int)
            # s_ind = np.ceil( ((s - self.state_low) / self.tile_width) + tiling_low).astype(int) 
            s_ind = np.ceil(((s - self.state_low) / (self.tiles_per_dim)) + tiling_low).astype(int) # - 1 
            for i in range(len(s_ind)):
                if s_ind[i] < 0:
                    s_ind[i] = 0
            tile_ind = np.ravel_multi_index(multi_index=s_ind, dims=self.tiles_per_dim)
            feature_vec[tiling_ind][tile_ind] = 1

        return feature_vec

    # update for linear function approx. on page 205
    def update(self,alpha,G,s_tau):
        self.w = self.w + alpha * (( G - self.__call__(s=s_tau) ) * self.get_feature_vec(s=s_tau))
        return self.w
