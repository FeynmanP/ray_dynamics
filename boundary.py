#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'You Mengyu'


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative


def covert(x, y):
    r = np.sqrt(x**2 + y**2)
    if x < 0:
        theta = np.arctan(y / x) + np.pi
    else:
        theta = np.arctan(y / x)

    return r, theta


class BdryPolar(object):
    """Parentobject for those cavity whose boundary can be represent by a function in parlor coordination."""
    def __init__(self):
        self.compute_bdry_data()

    def r(self, theta):
        """
        Dedine your own boundary equation.
        The only place should be changed in different cavity.
        """
        return 1

    def if_bdry(self, x, y):
        r, theta = covert(x, y)
        if r > self.r(theta):
            return True
        else:
            return False

    def formal_der_bdry(self, x, y):
        r, theta = covert(x, y)
        dr_dtheta = derivative(self.r, theta, 10e-6)
        tang = (np.tan(theta) * dr_dtheta + r) / (dr_dtheta - np.tan(theta) * r)
        return -1 / tang

    def compute_bdry_data(self):
        data = []
        for theta in np.linspace(0, 2*np.pi, 1000):
            r = self.r(theta)
            data.append([r * np.cos(theta), r * np.sin(theta)])

        self.bdry_data = np.array(data).transpose()


class Circle(BdryPolar):
    def __init__(self):
        super(Circle, self).__init__()

    def r(self, theta):
        return 1


class Quadropular1(BdryPolar):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(Quadropular1, self).__init__()

    def r(self, theta):
        e = self.epsilon
        return (1 / (1 + e**2/2)**(1/2)) * (1 + e * np.cos(2 * theta))

    
class Quadropular2(BdryPolar):
    def __init__(self, pars):
        self.r0, self.epsilon = pars
        super(Quadropular2, self).__init__()
        
    def r(self, theta):
        return self.r0 * (1 - self.epsilon * np.cos(2 * theta))
    
    

