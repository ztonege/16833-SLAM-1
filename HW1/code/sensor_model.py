'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 2

        self._norm_wts = 1.0
        
        self._map = occupancy_map

    
    def p_hit(self, z_star):
        
        return np.random.normal(z_star, self._sigma_hit)
    
    def p_short(self, z_star, z_t):
        
        if(z_star <= z_t):        
            return np.random.exponential(1/self._lambda_short)
        
        else:
            return 0
        
    def p_max(self, z_t):
    
        if(z_t >= 0.95*self._max_range and z_t <= 1.05*self._max_range):
            return 1
        else:
            return 0
        
        
    def p_rand(self, z_t):
        if(z_t < self._max_range and z_t >= 0):
           return 1/self._max_range
        
        else:
            return 0
        
    
    #https://www.codegrepper.com/code-examples/python/python+bresenham+line+algorithm
    def bresenham(self, x1, y1, x2, y2):
        
        dx = x2 - x1
        dy = y2 - y1
        
        if (abs(dy) > abs(dx)):
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
            
        dx = x2 - x1
        dy = y2 - y1
        
        error = int(dx /2.0)
        ystep = 1 if y1 < y2 else -1
        
        y = y1
        pointsx = []
        pointsy = []
        
        for x in range(x1, x2 + 1):
            
            if (abs(dy) > abs(dx)):
                coordx = y
                coordy = x
                
            else:
                coordx = x
                coordy = y
                
            pointsx.append(coordx)
            pointsy.append(coordy)
            
            m = self._map[coordx, coordy]
            
            prob = np.random.choices([0,1], weights = (100*(1-m), 100*m), k = 1)
            
            if(prob == 1):
                distance = np.sqrt((coordx - x1)**2 + (coordy - y1)**2)
                return distance
                break
            else:
                continue
            
                
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
                
            if swapped:
                pointsx.reverse()
                pointsy.reverse()
                
        return self._max_range/10
    
    
    def endpoint(self, x1, y1, theta1, k):
        
        x2 = x1 + self._max_range/10 * np.cos(theta1 - np.pi/2 + (k-1)*np.pi/180)
        y2 = y1 + self._max_range/10 * np.sin(theta1 - np.pi/2 + (k-1)*np.pi/180)
        
        return x2, y2
        

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        
        q = 1
        K = z_t1_arr.size
        
        zhit = self._z_hit
        zshort = self._z_short
        zmax = self._z_max
        zrand = self._z_rand
        
        for k in range(1, K+1):
            z_t = z_t1_arr[k]
            x1, y1, theta1 =  x_t1[0], x_t1[1], x_t1[2]
            x2, y2 = self.endpoint(x1, y1, theta1, k)
            z_star = self.bresenham(x1, y1, x2, y2)
            
            phit = self.phit(z_star)
            pshort = self.p_short(z_star, z_t)
            pmax = self.p_max(z_t)
            prand = self.p_rand(z_t)
            
            p = zhit*phit + zshort*pshort + zmax*pmax + zrand*prand
            
            q = q*p
                
        prob_zt1 = q
        return prob_zt1






















