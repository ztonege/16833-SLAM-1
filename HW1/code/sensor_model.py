'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
# import cupy as cp
import numpy as np
import math
import time
import random
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
        self._z_hit = 100#300 #1
        self._z_short = 5#50 #0.1
        self._z_max = 0.02#0.15 #0.1
        self._z_rand = 6000 #6000 #100

        self._sigma_hit =500#600 #50
        self._lambda_short = 0.001 #0.0006 #0.1

        self._max_range = 8183 #8183 #1000
        self._min_probability = 0.35 #0.35
        self._subsampling = 30

        self._norm_wts = 1.0
        
        self._map = occupancy_map
        self._end_points = []
        
    
    def p_hit(self, z_star, z_t):
        
        if(0 <= z_t <= self._max_range):
            normal = (1/np.sqrt(2*np.pi*(self._sigma_hit)**2))*np.exp(-0.5*(z_t - z_star)**2/(self._sigma_hit)**2)
            # return np.random.normal(z_star, self._sigma_hit)
            return normal
        else:
            return 0
        
    def p_short(self, z_star, z_t):
        
        if(0 <= z_t <= z_star):
            
            return self._lambda_short*(np.exp(-self._lambda_short*z_t))
            
            # return np.random.exponential(1/self._lambda_short)
        
        else:
            return 0
        
    def p_max(self, z_t):
    
        if(z_t >= 0.77*self._max_range and z_t <=self._max_range): #jugaad might have to fix
            return 1
        else:
            return 0
        
        
    def p_rand(self, z_t):
        if(0 <= z_t < self._max_range):
           return 1/self._max_range
        
        else:
            return 0
        
    
    #https://www.codegrepper.com/code-examples/python/python+bresenham+line+algorithm

    def true_range(self, points):
              
        
        x1, y1 = points[0]
        e_x, e_y = x1, y1
        # distance = 0
        for i in range(1,len(points)):
            x, y = points[i]
            # x = np.int(np.round(x/10.0))
            # y = np.int(np.round(y/10.0))
            # print("x, y", x, y)
            
            # if(x >= 800 or y >= 800):
            #     x = 799
            #     y = 799
                
            m = self._map[x, y]
            # print("m", m)

            prob = random.choices([0,1], weights = (100*(1-m), 100*m), k = 1)
            
            # print("prob", prob)
            # print(m, prob)
            


            if(prob[0] == 1 or m >= self._min_probability or m == -1):
            # if(m >= 0.75 or m == -1):
                e_x, e_y = x, y
                distance = 10*np.sqrt((x - x1)**2 + (y - y1)**2)
                
                break
            # if(m == -1): #Might not be needed
            #     distance = 0


            if(i == len(points)-1):
                e_x, e_y = x, y
                distance = self._max_range
            else:
                continue

        self._end_points.append((e_x, e_y))    
        
        return e_x, e_y, distance
        
           
    
    def bresenham(self, x1, y1, x2, y2):
        
        x1 = np.int(np.round(x1))
        y1 = np.int(np.round(y1))
        x2 = np.int(np.round(x2))
        y2 = np.int(np.round(y2))
        
        # x1 = np.int(x1)
        # y1 = np.int(y1)
        # x2 = np.int(x2)
        # y2 = np.int(y2)
        
        dx = x2 - x1
        dy = y2 - y1
        
        slopedir = abs(dy) - abs(dx)
        
        if (slopedir > 0):
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
        # pointsx = []
        points = []
        
        for x in range(x1, x2 + 1):
            
            if (slopedir > 0):
                coord = (y,x)
                # coord = x
                
            else:
                coord = (x,y)
                # coordy = y
                
            points.append(coord)
            # pointsy.append(coordy)
  
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
                
        if swapped:
            points.reverse()
            # pointsy.reverse()
                
        return points
    
    
    def endpoint(self, x1, y1, theta1, k):
        
        #Assuming theta1 faces straight, we go from -90 to 90
        x2 = x1 + self._max_range/10 * np.cos(theta1 - np.pi/2 + (k*self._subsampling)*np.pi/180) #removed -1
        y2 = y1 + self._max_range/10 * np.sin(theta1 - np.pi/2 + (k*self._subsampling)*np.pi/180)
        
        return x2, y2
        
    def beam_finder(self, args):
        
        z_t1_arr, x_t1 = args
        q = 0
        K = z_t1_arr.size
        
        x_t1_L = x_t1[0] + 25.0*np.cos(x_t1[2])
        y_t1_L = x_t1[1] + 25.0*np.sin(x_t1[2])
        
        x_t1_L /= 10.0
        y_t1_L /= 10.0
        
        zhit = self._z_hit
        zshort = self._z_short
        zmax = self._z_max
        zrand = self._z_rand
        
        
        x1, y1, theta1 =  x_t1_L, y_t1_L, x_t1[2]
        for k in range(1, K+1): #K = 180
            z_t = z_t1_arr[k-1] 
            # x1, y1, theta1 =  x_t1_L, y_t1_L, x_t1[2]
            x2, y2 = self.endpoint(x1, y1, theta1, k)
            
            points = self.bresenham(x1, y1, x2, y2)
            #if too slow, try formulating look up table
            #bresenham and endpoint outside for loop, have to shift center of circle, and pick half circle accordingly
            
            z_star = self.true_range(points)
            
            phit = self.p_hit(z_star, z_t)
            pshort = self.p_short(z_star, z_t)
            pmax = self.p_max(z_t)
            prand = self.p_rand(z_t)
            
            
            p = zhit*phit + zshort*pshort + zmax*pmax + zrand*prand
            if(p > 0):
                q = q + np.log(p)
                
        # prob_zt1 = q
        print("q =" , np.exp(q))
        return np.exp(q)

    def ray_viz(self, ray_pt):
        s_x, s_y = np.int(np.round(ray_pt[0][0])), np.int(np.round(ray_pt[0][1]))
        ray = []
        # print(type(ray))
        for i in range(1, ray_pt.shape[0]):
            e_x, e_y = ray_pt[i][0], ray_pt[i][1]
            # print(s_x, s_y,e_x, e_y)
            plt.plot([s_y, e_y], [s_x, e_x], c='r')
            # points = self.bresenham(s_x, s_y, e_x, e_y)
            # ray.append(list(points))
            # print(ray[0][0])
        # plt.show()
        # print("Ray shape =", (np.array(ray).shape))



    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # args = []
        # args.append((z_t1_arr,x_t1))
        
        # p = Pool(10)
        
        # q = p.map(self.beam_finder, args)
      
                
        # p.close()
        # p.join()
      
            
        # print("q =", q)  
        # return q

        
        q = 1
        K = np.int(z_t1_arr.size/self._subsampling)
        # print("The value of K is =", K) 
        
        x_t1_L = x_t1[0] + 25.0*np.cos(x_t1[2])
        y_t1_L = x_t1[1] + 25.0*np.sin(x_t1[2])
        
        x_t1_L /= 10.0
        y_t1_L /= 10.0
        
        zhit = self._z_hit
        zshort = self._z_short
        zmax = self._z_max
        zrand = self._z_rand
    
        x1, y1, theta1 =  x_t1_L, y_t1_L, x_t1[2]
        pts = np.array([x1, y1]).reshape(1, -1)

        for k in range(1, K+1): #K = 36
            z_t = z_t1_arr[k*self._subsampling-1] 
            # x1, y1, theta1 =  x_t1_L, y_t1_L, x_t1[2]
            x2, y2 = self.endpoint(x1, y1, theta1, k)
            
            points = self.bresenham(x1, y1, x2, y2)
            #if too slow, try formulating look up table
            #bresenham and endpoint outside for loop, have to shift center of circle, and pick half circle accordingly
            
            e_x, e_y, z_star = self.true_range(points)
            end_pt = np.array([e_x, e_y]).reshape(1, -1)
            pts = np.append(pts, end_pt, axis = 0)
            # print("Endpoints = ", e_x, e_y)
            # print("Starting point =", np.int(np.round(x1)), np.int(np.round(y1)))
            # print("Distance =", z_star)
            # print("end pt", end_pt)
            
            phit = self.p_hit(z_star, z_t)
            pshort = self.p_short(z_star, z_t)
            pmax = self.p_max(z_t)
            prand = self.p_rand(z_t)
            
            
            p = zhit*phit + zshort*pshort + zmax*pmax + zrand*prand
            if(p > 0):
                q = q*p
                
        # print("Endpoints = ", pts.shape) #(19, 2)
        # print("Starting point =", x1, y1)
        # print("Distance =", z_star)

        self.ray_viz(pts)
        # prob_zt1 = q
        # print("q =", q)  
        return q
    

    
if __name__ == "__main__":
    
    # pass
    path_map ='../data/map/wean.dat'
    map_obj = MapReader(path_map)
    occupancy_map = map_obj.get_map()
        
    sensor = SensorModel(occupancy_map)
        
    # points = sensor.bresenham(200,500, 100, 600)
    # distance = sensor.true_range(points)
    # print(points)
    # print(distance)
    z_star = 2000
    z_t = np.arange(sensor._max_range)
    p_t = []
    zhit = sensor._z_hit
    zshort = sensor._z_short
    zmax = sensor._z_max
    zrand = sensor._z_rand
    for i in range(sensor._max_range):
        phit = sensor.p_hit(z_star, z_t[i])
        pshort = sensor.p_short(z_star, z_t[i])
        pmax = sensor.p_max(z_t[i])
        prand = sensor.p_rand(z_t[i])
            
        p = zhit*phit + zshort*pshort + zmax*pmax + zrand*prand
        p_t.append(p)   
    

    p_t = np.asarray(p_t)
    
    plt.figure()
    plt.scatter(z_t, p_t, s = 1)



















