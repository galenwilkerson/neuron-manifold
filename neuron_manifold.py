__author__      = "galen wilkerson"

import cv2
from mat4py import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_mat(filename):
    '''
    load the .mat (array) file and return as numpy array
    '''
    
    mask_data = loadmat(filename)
    mask = mask_data.values()
    baseline_1_mask = np.array(list(mask))[0,:,:]
    return(baseline_1_mask)

def find_center(this_neuron_mask):
    '''
    given a binary image as a 2-d numpy array, return the coordinates of its center
    '''
    
    # calculate moments of binary image
    M = cv2.moments(this_neuron_mask.astype(np.uint8))

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    return(cX, cY)

def find_neuron_centers(baseline_1_mask):
    '''
    given a numpy array of neuron masks 
    (where there is a unique id in a contiguous blob for each neuron)
    
    return a list of x,y coordinates for the center of each neuron (blob)
    '''
    
    # list of unique neurons
    unique_neurons = np.unique(baseline_1_mask)
    unique_neurons = unique_neurons[1:]

    coords = []

    # only look at each neuron
    for neuron_id in unique_neurons:
        this_neuron_mask = (baseline_1_mask == neuron_id)
        coords.append(find_center(this_neuron_mask))
        
    return np.array(coords)

def draw_neuron_centers(baseline_1_mask, coords):
    '''
    draw the neurons and their centers
    '''
    
    plt.figure()

    for (cX, cY) in coords:
        plt.plot(cX, cY, marker=".", markersize=2, color = 'red')

    sns.heatmap(baseline_1_mask > 0);


def get_binarized_firing_at_timestep(baseline_deconv, time_step, firing_threshold = 0):
    # neuron firing at time_step
    firing_at_timestep = baseline_deconv[time_step,:]

    binarized_firing = np.array(firing_at_timestep > firing_threshold)
    
    return(binarized_firing) 


def get_binarized_firing_array(baseline_deconv, firing_threshold = 0):
    '''
    input:  numpy array of all node firing
    return array where 1 = firing, 0 = not firing
    '''

    binarized_firing = (baseline_deconv > firing_threshold).astype(int)

    return(binarized_firing)


