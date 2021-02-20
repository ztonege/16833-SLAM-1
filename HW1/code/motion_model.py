'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.01
        self._alpha4 = 0.01

        

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        x_t1 = np.zeros(3)
        
        alpha1 = self._alpha1
        alpha2 = self._alpha2
        alpha3 = self._alpha3
        alpha4 = self._alpha4
        
        drot1 = np.arctan2(u_t1[1] - u_t0[1],u_t1[0] - u_t0[0]) - u_t0[2]
        dtrans = np.sqrt((u_t1[1] - u_t0[1])**2+(u_t1[0] - u_t0[0])**2)
        drot2 = u_t1[2] - u_t0[2] - drot1
        
        drot1_ = drot1 - np.random.normal(0, np.sqrt(alpha1*drot1**2 + alpha2*dtrans**2))
        dtrans_ = dtrans - np.random.normal(0, np.sqrt(alpha3*dtrans**2 + alpha4*drot1**2 + alpha4*drot2**2))
        drot2_ = drot2 - np.random.normal(0, np.sqrt(alpha1*drot2**2 + alpha2*dtrans**2))
        
        x_t1[0] = x_t0[0] + dtrans_*np.cos(x_t0[2] + drot1_)
        x_t1[1] = x_t0[1] + dtrans_*np.sin(x_t0[2] + drot1_)
        x_t1[2] = x_t0[2] + drot1_ + drot2_
        
        return x_t1
        
