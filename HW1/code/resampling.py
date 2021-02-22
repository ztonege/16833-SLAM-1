'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import random


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        
        X_bar_resampled =  np.zeros_like(X_bar)
        
        wts = X_bar[:,3]
        
        freq = np.random.multinomial(num_particles, wts/np.sum(wts))
        print(freq)
        a = 0
        # b = 0
        # count = 0
        for i in range(num_particles):
  
            # if(freq[i] > 0):
                
            #     X_bar_resampled[b] = X_bar[count]
            
            #     a += 1
            #     b += 1
            #     if(a > freq[i]):
            #         a = 0
            #         count += 1
                    
            # elif(freq[i] == 0):
            #         count += 1
            
            resampled = []
            
            if(freq[i] > 0):
                arr = np.tile(X_bar[i], (freq[i], 1))
                # print(arr.shape)
                X_bar_resampled[a: a + freq[i]] = arr
                
                a += freq[i]
                # print("a =", a)
                # print("    ")
       
        X_bar_resampled[:,3] = np.ones(num_particles).reshape(1, num_particles)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        
        X_bar_resampled =  np.zeros_like(X_bar)
        
        r = random.uniform(0, 1/num_particles)
        w = X_bar[:,3]/np.sum(X_bar[:,3])
        
        c = w[0]
        i = 0
        a = 0
        for j in range(num_particles):
            U = r + j*(1/num_particles)
            while(U > c):
                i += 1
                # print("i = ",i)
                c += w[i]
                
            X_bar_resampled[a] = X_bar[i]
            a+=1
        
        X_bar_resampled[:,3] = np.ones(num_particles).reshape(1, num_particles)
        
        return X_bar_resampled
    
    
if __name__ == "__main__":
    pass

    # sampler = Resampling()
    
    # X_bar = (np.arange(40)).reshape(10,4)
    # X_bar[9:10, 3] = 500
    # print(X_bar)
    
    # X_bar_resampled = sampler.low_variance_sampler(X_bar)
    # print(X_bar_resampled)
    
