#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:23:37 2019

@author: zhechuang
"""

#"numGoodSaled"

import numpy as np
import pandas as pd

class NewsVendorGame:
    
    def __init__(self,K, Kpr, Kst, Kpe, Ktr, CWarehouse, CTruck, Price, dmax):
        """
        K:      int, number of warehouses
        Kpr:    float,          unit product cost parameter
        Kst:    array of float, storage cost parameter
        Kpe:    float,          penalty cost parameter
        Ktr:    array of float, transportation cost parameter
        CWarehouse:    array of float, storage capacity 
        CTruck: array of float  , truck capacity, length is K+1, with the first being np.nan
        Price:  float, price of one unit of product
        dmax:   float, max demand
        """
        self.K=K
        self.Kpr=Kpr
        self.Kst=Kst
        self.Kpe=Kpe
        self.Ktr=Ktr 
        self.CWarehouse=CWarehouse
        self.CTruck=CTruck
        self.Price=Price
        self.dmax= dmax
        
        assert len(self.CWarehouse)==self.K+1, "warehouse capacity, length is K+1"
        assert len(self.CTruck)==self.K+1, "truck capacity, length is K+1"
        self.CTruck[0]=np.nan
        
        
        self.t=0  #timestamp
        self.stock=np.zeros(K+1) 
        self.old_d=np.zeros(K+1) #d(t-1), but d[0] is always NA
        self.old_d[0]=np.nan
        self.d=np.zeros(K+1) #d(t)
        self.d[0]=np.nan
        #self.state=(self.stock, self.old_d, self.d)
        
       
        
    def update_demand(self):
        j = np.array(range(0,self.K+1))
        demand = np.floor(self.dmax/2 * np.sin(2*np.pi*(self.t + 2 * j)/12) + self.dmax/2 + np.random.choice([0,1]))
        demand[0] = np.nan
        return demand

    
    def human_interactions(self):
        print('current time:',self.t)
        print('current storage:',self.stock)
        print('last demand:',self.d)
        print('please input your decision for a0, a1,.., aK, seperated in comma:')
        assignment=np.array(list(map(float,input().split(',')))).reshape(self.K+1)
        (stock, d, old_d), profit=self.step_game(assignment)
        print('-----')
        print('profit:', profit)
        print('next storage:', stock)
        print('d:', d)
        print('old d:', old_d)
        print('################')
        

    def transition_stock(self, stock, demand, assignment):
        #new_stock = np.zeros(len(stock))
        stock[0] = min(stock[0]+assignment[0] -
                       np.sum(assignment[1:]), self.CWarehouse[0])
        stock[1:] = np.minimum(stock[1:] + assignment[1:] - demand[1:], self.CWarehouse[1:])
        return stock

    def reward(self, stock, demand, assignment):
        """
        calculate the reward
        demand:  demand of current time t
        assignment: assignment of current time t
        """
        r = (self.Price * np.sum(demand[1:]) - 
        self.Kpr*assignment[0] - 
        np.dot(self.Kst,np.maximum(stock,0)) + 
        self.Kpe*np.sum(np.minimum(stock[1:],0)) - 
        np.sum(self.Ktr[1:]* np.ceil(assignment[1:]/self.CTruck[1:])))
        print(self.Price * np.sum(demand[1:]))
        print(Kpr*assignment[0])
        print(np.dot(self.Kst,np.maximum(stock,0)))
        print(Kpe*np.sum(np.minimum(stock[1:],0)) )
        print(np.sum(Ktr[1:]* np.ceil(assignment[1:]/self.CTruck[1:])))
        print(r)
        return r
    
    
    
    def step_game(self, assignment): 
        self.old_d = self.d
        self.d = self.update_demand()
        profit = self.reward(self.stock,self.d, assignment)
        self.stock = self.transition_stock(self.stock, self.d, assignment)
        self.t=self.t+1
        return (self.stock, self.d, self.old_d), profit

    
    
if __name__=='__main__':
    
    K=2
    Kpr=5
    Kst=np.array([0,0,0]).reshape(K+1)
    Kpe=5
    Ktr=np.array([np.nan,0,0]).reshape(K+1)
    CWarehouse=np.array([20,20,20]).reshape(K+1)
    CTruck= np.array([np.nan,10,10]).reshape(K+1)
    Price=10
    dmax= 3
    
    
    game=NewsVendorGame(K, Kpr, Kst, Kpe, Ktr, CWarehouse, CTruck, Price, dmax)
    
    for time in range(3):
        game.human_interactions()
    
    
    