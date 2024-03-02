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
        self.w = np.zeros(num_tilings)
        self.values
        # TODO: implement this method

    def __call__(self,s):
        # TODO: implement this method
        # this is v(s, w)
        # inner product of feature vector (tiles) for state s and w 
        return 0.

    # update for linear function approx. on page 205
    def update(self,alpha,G,s_tau):
        self.w = self.w + alpha * (( G - self.__call__(s=s_tau) ) * self.values[s_tau])
        return self.w
