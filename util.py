'''
This file contains useful utility functions.
'''
import numpy as np

def angle_between(pos1, pos2):
    ''' Computes the angle between two positions. '''
    diff = pos2 - pos1
    return np.arctan2(diff[1], diff[0])

def angle_position(theta):
    ''' Computes the position on a unit circle at angle theta. '''
    return vector(np.cos(theta), np.sin(theta))

def vector(xvalue, yvalue):
    ''' Returns a 2D numpy vector. '''
    return np.array([float(xvalue), float(yvalue)])

def vector_to_tuple(vect):
    ''' Converts a 2D vector to a tuple. '''
    return (vect[0], vect[1])

def to_matrix(vect):
    ''' Turns a vector into a single column matrix. '''
    return np.array([vect]).T
