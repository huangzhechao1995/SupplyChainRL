import numpy as np


def transition_stock(self, stock, demand, assignment):
    new_stock = np.zeros(len(stock))
    stock[0] = min(stock[0]+assignment[0] -
                   np.sum(assignment[1:]), self.CWarehouse[0])
    stock[1:] = np.minimum(stock[1:] + assignment[1:] - demand[1:], self.capacity=0), self.CWarehouse[1:])
    return stock

def reward(self, stock, demand, assignment):
    r = (self.Price * np.sum(demand[1:]) - 
    Kpr*assignment[0] - 
    np.sum(np.maximum(stock,0)) + 
    Kpe*np.sum(np.minimum(stock[1:],0)) - 
    np.sum(Ktr[1:]* np.ceil(assignment[1:]/self.CTruck))0
    return r



def step_game(self, assignment): 
    self.old_d = self.d
    self.d = self.update_demand()
    profit = self.reward(self.stock,self.d, assignment)
    self.stock = self.transition_stock(self.stock, self)    
    return (self.stock, self.d, self.old_d, profit)