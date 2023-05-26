#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'You Mengyu'
"""
Boundary of cavity should be defined here.
For the cavities which do not have continuous boundary,
"""
import numpy as np
from scipy.misc import derivative
import scipy.integrate as integrate
import inspect
import sys
import absorber


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
        try:
            self.compute_bdry_data()
        except TypeError:
            print('Program can not be excuted.\nCheck the parameters and the type of cavity')
            exit()

    def bdry(self, phi):
        """ Different with different cavities"""
        return 1

    def __str__(self):
        """Define cavity name"""
        return 'Cavity name'

    def is_inside(self, x, y):
        r, phi = covert(x, y)
        return r < self.bdry(phi)

    def dr_dphi(self, phi):
        return derivative(self.bdry, phi, 10e-6)

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

    def leak(self, phi, theta):
        return 1

    def is_tailor(self, x, y):
        return 0
'''
    def convert_phi_s(self, end, start=-np.pi):
        """0 < phi <= 2*pi"""
        length = integrate.quad(lambda x: np.sqrt(self.dr_dphi(x)**2 + self.bdry(x)**2), -np.pi, np.pi)
        result = integrate.quad(lambda x: np.sqrt(self.dr_dphi(x)**2 + self.bdry(x)**2), start, end)

        frac = result[0] / length[0]
        return frac

    def convert_s_phi(self, s):
        return s
'''

class CircleXY:

    def __init__(self, pars):
        self.R = pars

    def is_inside(self, x, y):
        return x**2 + y**2 <= self.R**2

    def normal_vector_bdry(self, x, y):
        if x > 0 and y > 0:
            return self.quater_oval_norm_vector(x, y)
        elif x < 0 and y > 0:
            return self.quater_oval_norm_vector(-x, y)

    def quater_oval_norm_vector(self, x, y):
        pass


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
        self.e, self.epsilon = pars
        super(Ellipse, self).__init__()

    def __str__(self):
        return 'Ellipse'

    def bdry(self, phi):
        return self.epsilon / (1 - self.e * np.cos(phi))


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
        return 'Quadropular2'

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
        self.theta = np.arccos(self.d / self.r)
        self.length_line = 2 * np.sqrt(self.r ** 2 - self.d ** 2)
        self.length = self.r * (2*np.pi - 2 * self.theta) + self.length_line
        self.frac1 = 1/2 * self.length_line / self.length
        self.frac2 = 1/2
        self.frac3 = 1 - self.frac1
        super(D, self).__init__()

    def bdry(self, phi):
        theta = np.arccos(self.d/self.r)
        if -theta < phi < theta:
            return self.d / np.cos(phi)
        else:
            return self.r

    def convert_phi_s(self, phi):
        theta = np.arccos(self.d / self.r)
        length_line = 2 * np.sqrt(self.r**2 - self.d**2)
        length = self.r * (2*np.pi - 2*theta) + length_line
        if 0 <= phi < theta:
            return (self.d * np.abs(np.tan(phi))) / length
        elif -theta < phi < 0:
            return 1 - (self.d * np.abs(np.tan(phi))) / length
        elif phi >= theta:
            return ((phi - theta) * self.r + 1/2 * length_line) / length
        else:
            return ((2 * np.pi + phi - theta) * self.r + 1/2 * length_line) / length

    def convert_s_phi(self, s):
        theta = np.arccos(self.d / self.r)
        length_line = 2 * np.sqrt(self.r ** 2 - self.d ** 2)
        length = self.r * (2*np.pi - 2 * theta) + length_line
        frac1 = 1/2 * length_line / length
        frac2 = 1/2
        frac3 = 1 - frac1
        if s < frac1:
            return np.arctan(length * s / self.d)
        elif frac1 <= s < frac2:
            return (s * length - 1/2 * length_line) / self.r + theta
        elif frac2 <= s < frac3:
            return (s * length - 1/2 * length_line) / self.r + theta - 2 * np.pi
        else:
            return - np.arctan((1 - s) * length / self.d)

    def full_leak(self, phi, theta):
        ds = 0.5
        d_phi = np.arctan(ds / self.d)
        if 0 < phi < d_phi:
            return 0
        else:
            return 1

    def partial_leak_TM(self, phi, theta):
        n = 3.3
        theta2 = np.arcsin(n * np.sin(theta))
        if np.sin(theta) <= 1/n:
            relectivity = (np.sin(theta - theta2) / np.sin(theta + theta2)) ** 2
            return relectivity
        else:
            return 1

    def leak(self, phi, theta):
        return self.partial_leak_TM(phi, theta)

    def boundary_vector(self, s):
        """
        :param phi: angle in porlar coordinate:
        :param s: fraction of the arc length from 0
        :return: clockwise boundary vectors
        """
        phi = self.convert_s_phi(s)

        if s < self.frac1:
            return np.array([0, 1])
        elif self.frac3 >= s >= self.frac1:
            return np.array([-np.sin(phi), np.cos(phi)])
        else:
            return np.array([0,1])

    def is_tailor_v1(self, x, y):
        #  Tailer area to adujest the phase space, mainly effects the 3-period orbits in D-shapled cavity.
        #  v1.0 can only build a square absorber whose lines are parallel to x and y-axis.
        xmin = -0.83
        xmax = -0.63
        ymin = -0.01
        ymax = 0.01
        x_area = [xmin, xmax]
        y_area = [ymin, ymax]
        # x_area = [-1, 0]
        # y_area = [-0.4, 0.4]
        if x_area[0] <= x <= x_area[1] and y_area[0] <= y <= y_area[1]:
            return 1
        else:
            return 0

    def is_tailor(self, x, y):
        self.is_tailor_v1(x, y)


    def __str__(self):
        return 'D-shape'


class Oval(BdryPolar):
    def __init__(self, pars):
        self.r0, self.epsilon1, self.epsilon2, self.epsilon3, self.epsilon4 = pars
        super(Oval, self).__init__()

    def bdry(self, phi):
        r0 = self.r0
        e1, e2, e3, e4 = self.epsilon1, self.epsilon2, self.epsilon3, self.epsilon4
        r = r0 * (1 + e1 * np.cos(2*phi) + e2 * np.cos(4*phi) + e3 * np.cos(6*phi) + e4 * np.cos(8*phi))
        return r

    def __str__(self):
        return 'Oval'


class BoundaryXY:
    def __init__(self):
        self.bdry_data = []
        try:
            self.compute_bdry_data()
        except TypeError:
            print('Program can not be excuted.\nCheck the parameters and the type of cavity')
            exit()

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
        return -1

    def compute_bdry_data(self):
        for phi in np.linspace(-np.pi, np.pi, 10000):
            r = self.bdry(phi)
            self.bdry_data.append([r * np.cos(phi), r * np.sin(phi)])


class Stadium(BdryPolar):
    """This cavity is defined in x-y cooridnates"""
    def __init__(self, r):
        self.R = r
        super(Stadium, self).__init__()

    def bdry(self, phi):
        if -1/4 * np.pi < phi <= 1/4 * np.pi:
            return 2 * self.R * np.cos(phi)
        elif 1/4 * np.pi < phi <= 1/2 * np.pi:
            return self.R / np.sin(phi)
        elif 1/2 * np.pi < phi <= 3/4 * np.pi:
            return self.R / np.sin(phi)
        elif phi > 3/4 * np.pi:
            return -2 * self.R * np.cos(phi)
        elif phi <= -3/4 * np.pi:
            return -2 * self.R * np.cos(phi)
        elif - 3/4 * np.pi < phi <= -1/2 * np.pi:
            return -self.R / np.sin(phi)
        elif - 1/2 * np.pi < phi <= -1/4 * np.pi:
            return -self.R / np.sin(phi)
        else:
            return -2 * self.R * np.cos(phi)


class Lemon(BdryPolar):
    def __init__(self, l):
        self.L = l
        super(Lemon, self).__init__()

    def bdry(self, phi):
        if 0 < phi < 1/2 * np.pi:
            a = 1 + np.tan(phi)**2
            b = 2 * (1 - self.L / 2)
            c = (1 - self.L/2) ** 2 - 1
            value1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            x = value1
        elif phi == 1/2 * np.pi or phi == -1/2 * np.pi:
            return self.L - self.L**2 / 4
        else:
            a = 1 + np.tan(phi) ** 2
            b = - 2 * (1 - self.L / 2)
            c = self.L * (self.L/4 - 1)
            value1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            x = value1
        return np.sqrt((1 + np.tan(phi)**2) * x**2)


class Cardioid(BdryPolar):
    def __init__(self, pars):
        self.r0 = pars
        super(Cardioid, self).__init__()

    def __str__(self):
        return 'FlattenedQuadropule'

    def bdry(self, phi):
        return self.r0 * np.sqrt(1 + np.cos(phi))


class Mushroom(BdryPolar):
    def __init__(self, pars):
        """a < R and l, R > 0.1"""
        self.R, self.l, self.a = pars
        super(Mushroom, self).__init__()

    def __str__(self):
        return 'Mushroom'

    def bdry(self, phi):
        d = 0.5
        R = self.R
        l = self.l
        a = self.a

        phi1 = np.arctan2(-d, R-d)
        phi2 = np.arctan2(R-d, -d)
        phi3 = np.arctan2(-d-l, -d)
        phi4 = np.arctan2(-d-l, a-d)
        phi5 = np.arctan2(-d, a-d)

        if phi1 < phi <= phi2:
            return np.sqrt(R**2 - d**2 * (1 - 2 * np.cos(phi) * np.sin(phi))) - d * (np.cos(phi) + np.sin(phi))
        elif phi > phi2 or phi <= phi3:
            return np.abs(d / np.cos(phi))
        elif phi3 < phi <= phi4:
            return np.abs((l + d) / np.sin(phi))
        elif phi4 < phi <= phi5:
            return np.abs((a - d) / np.cos(phi))
        else:
            return np.abs(d / np.sin(phi))


def test():
    s = Mushroom((10, 10, 5))
    print(s.bdry(0))
    print(s.bdry(1/4 * np.pi))
    print(s.bdry(2/4 * np.pi))
    print(s.bdry(3/4 * np.pi))
    print(s.bdry(4/4 * np.pi))
    print(s.bdry(-1/4 * np.pi))
    print(s.bdry(-2/4 * np.pi))
    print(s.bdry(-3/4 * np.pi))
    print(s.bdry(-4/4 * np.pi))


def test2():
    c = D((5, 10))
    # for i in np.linspace(0, np.pi, 10):
    #    print(c.convert_phi_s(i))
    for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print((c.convert_s_phi(s)/(2*np.pi)) * 360, c.boundary_vector(s))


def print_classes():
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    return [i[0] for i in clsmembers]


if __name__ == '__main__':
    test2()





