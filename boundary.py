#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'You Mengyu'
"""
Boundary of cavity should be defined here.
For the cavities which do not have continuous boundary,
"""
import numpy as np
from scipy.misc import derivative


def covert(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return r, phi


e_x = np.array([1, 0])
e_y = np.array([0, 1])


def test():
    pass


class BdryPolar:
    """
    Parentclass for all the cavities whose boundary can be well-defined in polar coordinate
    """

    def __init__(self):
        self.bdry_data = []
        self.compute_bdry_data()

    def bdry(self, phi):
        """ Different with different cavities"""
        return 1

    def __str__(self):
        """Define cavity name"""
        return 'Cavity name'

    def is_inside(self, x, y):
        r, phi = covert(x, y)
        return r < self.bdry(phi)

    def normal_vector_bdry(self, x, y):
        """
        :return: normalized normal vector on boundary.
        """
        r, phi = covert(x, y)
        dr_dphi = derivative(self.bdry, phi, 10e-6)

        e_r = np.array([np.cos(phi), np.sin(phi)])
        e_phi = np.array([-np.sin(phi), np.cos(phi)])

        normal_vector = -self.bdry(phi) * e_r + dr_dphi * e_phi

        return normal_vector / np.linalg.norm(normal_vector)

    def compute_bdry_data(self):
        for phi in np.linspace(-np.pi, np.pi, 10000):
            r = self.bdry(phi)
            self.bdry_data.append([r * np.cos(phi), r * np.sin(phi)])


class Circle(BdryPolar):
    def __init__(self, pars):
        self.R = pars
        super(Circle, self).__init__()

    def bdry(self, phi):
        return self.R

    def __str__(self):
        return 'Circle'


class Ellipse(BdryPolar):
    def __init__(self, pars):
        self.e, self.p = pars
        super(Ellipse, self).__init__()

    def __str__(self):
        return 'Ellipse'

    def bdry(self, phi):
        return self.e * self.p / (1 - self.e * np.cos(phi))


class Quadropular1(BdryPolar):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(Quadropular1, self).__init__()

    def __str__(self):
        return 'Quadropular1'

    def bdry(self, phi):
        e = self.epsilon
        return (1 / (1 + e ** 2 / 2) ** (1 / 2)) * (1 + e * np.cos(2 * phi))


class Quadropular2(BdryPolar):
    def __init__(self, pars):
        (self.r0, self.epsilon) = pars
        super(Quadropular2, self).__init__()

    def __str__(self):
        return Quadropular2

    def bdry(self, phi):
        return self.r0 * (1 - self.epsilon * np.cos(2 * phi))


class FlattenedQuadropule(BdryPolar):
    def __init__(self, pars):
        (self.r0, self.epsilon) = pars
        super(FlattenedQuadropule, self).__init__()

    def __str__(self):
        return 'FlattenedQuadropule'

    def bdry(self, phi):
        return self.r0 * np.sqrt(1 + 2 * self.epsilon * np.cos(2 * phi))


class Bdry1(BdryPolar):
    """from "Ray chaos and Q-spoiling in lasing droplet" """

    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(Bdry1, self).__init__()

    def bdry(self, phi):
        return 1 + self.epsilon * ((np.cos(phi)) ** 2 + 1.5 * (np.cos(phi)) ** 4)


class D(BdryPolar):
    def __init__(self, pars):
        self.d, self.r = pars
        super(D, self).__init__()

    def bdry(self, phi):
        theta = np.arccos(self.d/self.r)
        if -theta < phi < theta:
            return self.d / np.cos(phi)
        else:
            return self.r




if __name__ == '__main__':
    test()
