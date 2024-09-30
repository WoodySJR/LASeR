# This file defines functions that transform matrix-formed and one-hot-encoded morphologies back and forth. 

import torch
import numpy as np
    
def operator(width):
    operator_2 = torch.zeros((5*width**2, width**2))
    for i in range(width**2):
        operator_2[5*i:5*(i+1),i] = torch.tensor([0,1,2,3,4])
    return operator_2
    
def vec_to_morph(robots, operator_2, width):
    robots = torch.mm(robots, operator_2)
    return robots.reshape(robots.shape[0], width, width)

def morph_to_vec(robots):
    bodies = robots.reshape(robots.shape[0],-1)
    robots = []
    for i in range(bodies.shape[0]):
        body = []
        for j in range(bodies.shape[1]):
            voxel = [0,0,0,0,0]
            voxel[int(bodies[i,j])]=1
            body += voxel
        robots.append(body)
    robots = torch.tensor(np.array(robots))
    return robots