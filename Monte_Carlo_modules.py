import numba 
import numpy as np
import random
import math 

from scipy import integrate, optimize, special, linalg
from scipy.sparse.linalg import eigs

import matplotlib.pylab as plt




@numba.jit(nopython=True, fastmath=True)
def H(x,y,z):
    return math.erf((x-y)/z) + math.erf((x+y)/z)

@numba.jit(nopython=True, fastmath=True)
def energ_func(w,k_new):
    return -np.log((H(k_new,w,100)-H(k_new,k_new,100))/(H(k_new,0,100)-H(k_new,k_new,100))) 

@numba.jit(nopython=True, fastmath=True)
def energ_func_adapt(w,k_new,d):
    return -np.log((H(k_new,w,d)-H(k_new,k_new,d))/(H(k_new,0,d)-H(k_new,k_new,d)))



########################################
#### Symmetric Perceptron Functions ####
########################################

##### No Chain (Single Monte-Carlo)
# This recompute efficiently w=ξ.x when one bit of x has been flipped
@numba.jit(nopython=True, fastmath=True)
def d_interaction_vector_func(interaction_vector,G,indice_flip,sign_flip):
    interaction_vector_new=np.zeros(len(interaction_vector))
    for i in range(len(interaction_vector)):
        interaction_vector_new[i]=interaction_vector[i]+2*sign_flip*G[i,indice_flip]
    return interaction_vector_new


# This compute the energy of a configuration: E=sum_{mu=1}^P Loss(ξ_{mu}.x) 
@numba.jit(nopython=True, fastmath=True)
def perceptron_cost(w_vec,k_new):
    
    out=0
    for k in range(len(w_vec)):
        cost=energ_func(w_vec[k],k_new) 
        out+=cost
    return out



##### Chain (Several Monte-Carlo)
# This compute the energy of a configuration (with the possibility to adapt the loss function with variable d ):  E=sum_{mu=1}^P Loss(ξ_{mu}.x)  
@numba.jit(nopython=True, fastmath=True)
def perceptron_cost_adapt_bias(w_vec,k_new,d):
    
    out=0
    for k in range(len(w_vec)):
        cost=energ_func_adapt(w_vec[k],k_new,d) 
        out+=cost
        
    return out

@numba.jit(nopython=True, fastmath=True)
def perceptron_cost_adapt_bias2(w_vec,k_new,d1,d2):

    out=0
    for k in range(len(w_vec)):
        cost=energ_func_adapt(w_vec[k],k_new,d1)+energ_func_adapt(w_vec[k],k_new,d2)
        out+=cost
        
    return out

@numba.jit(nopython=True, fastmath=True)
# This compute the energy of a configuration in the chain, if your are the 0^th element your energy is the usual one, 
# if your are any other element of the chain you get the energy with a bias d
def perceptron_cost_chain(w_vec,k_new,link_index,d):

    if link_index==0: 
        return perceptron_cost(w_vec,k_new)
    else:
        return perceptron_cost_adapt_bias(w_vec,k_new,d)
        
########################################
########################################







########################################
########### Chain Functions ############
########################################
# This recompute efficiently the interactions \sum_{i=1}^{M-1} h_i(x_i.x_{i+1}) when one bit flip has been perform on one of the configuration of the chain
@numba.jit(nopython=True, fastmath=True)
def d_field_loss(field_loss,x_vec,h,link_flip,indice_flip,sign_flip):
    M_=len(x_vec[:,0])
    field_loss_new=np.zeros(M_-1)
    for k in range(M_-1):
        field_loss_new[k]=field_loss[k]
        
    
    if link_flip==0:
        field_loss_new[0]          +=2*sign_flip*h[0]          *x_vec[1,indice_flip]
        
    if link_flip==M_-1:
        field_loss_new[M_-2]       +=2*sign_flip*h[M_-2]       *x_vec[M_-2,indice_flip]
        
    if link_flip!=0 and link_flip!=M_-1:
        field_loss_new[link_flip]  +=2*sign_flip*h[link_flip]  *x_vec[link_flip+1,indice_flip]
        field_loss_new[link_flip-1]+=2*sign_flip*h[link_flip-1]*x_vec[link_flip-1,indice_flip]
                         
        
    return field_loss_new


@numba.jit(nopython=True, fastmath=True)
# This compute the energy constraining the chain: \sum_{i=1}^{M-1} h_i(x_i.x_{i+1})
def field_loss_func(x_vec,h,link_index):  
    return h[link_index]*np.dot(x_vec[link_index,:],x_vec[link_index+1,:])

########################################
########################################


