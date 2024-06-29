#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'You Mengyu'
"""
Boundary of cavity should be defined here.
For the cavities which do not have continuous boundary,
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, arctan2, tan
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.optimize import brentq



def order_boundary_points_counterclockwise(boundary_points):
    """Order the points of boundary from 0 to 2pi"""
    # Compute the centroid of the boundary points
    centroid = np.mean(boundary_points, axis=0)

    # Compute the angles between each boundary point and the centroid
    angles = np.arctan2(boundary_points[:, 1] - centroid[1], boundary_points[:, 0] - centroid[0])

    # Adjust the angles to be in the range [0, 2pi]
    angles = (angles + 2*pi) % (2*pi)
    # Sort the boundary points based on these angles
    sorted_indices = np.argsort(angles)
    ordered_boundary_points = boundary_points[sorted_indices]

    return ordered_boundary_points


class BoundaryUniformPolar:
    """A general class for the cavity with uniform distribution of refractive index. Use Circle for instance"""
    def __init__(self, cavity_params):
        """
        Initialization.
        """
        self._set_cavity_params(cavity_params)
        self._generate_bdry_data()

    def __str__(self):
        return "CirclePolar"

    def _set_cavity_params(self, cavity_params):
        radius = cavity_params
        self.R = radius

    def _generate_bdry_data(self):
        values_phi = np.linspace(0, 2 * pi, 2048)
        values_x = [self.r_phi(phi) * cos(phi) for phi in values_phi]
        values_y = [self.r_phi(phi) * sin(phi) for phi in values_phi]
        values_pos_bdry = np.column_stack((values_x, values_y))
        self.bdry_data = values_pos_bdry

    def r_phi(self, phi):
        """Definition of the cavity shape in polar coordinates."""
        return self.R

    def is_inside(self, x, y):
        phi = arctan2(y, x)
        phi = (phi + 2*pi) % (2*pi)
        r = self.r_phi(phi)
        return (x**2 + y**2) < r**2

    # def perimeter(self):
    #     """Perimeter of the cavity."""
    #     return 2 * pi * self.R

    def perimeter(self):
        return self.compute_arc_length(2*pi)

    def dr_dphi(self, phi):
        # Calculate the derivative of r with respect to phi
        return derivative(self.r_phi, phi, dx=1e-6)

    def integrand(self, phi):
        # Calculate the integrand of the arc length formula
        r = self.r_phi(phi)
        dr_phi = self.dr_dphi(phi)
        return np.sqrt(r ** 2 + dr_phi ** 2)

    def compute_arc_length(self, phi1):
        # Compute the arc length from phi = 0 to phi = phi1
        arc_length, _ = quad(self.integrand, 0, phi1)
        return arc_length

    def arc_length_to_phi(self, s, phi_guess=2 * np.pi):
        # Function to find the difference between the computed arc length and the given s
        def func(phi):
            return quad(self.integrand, 0, phi)[0] - s

        # Use a root-finding algorithm to solve for phi
        phi_solution = brentq(func, 0, phi_guess)
        return phi_solution

    def convert_xy_to_s(self, x, y):
        """Convert the point (x, y) on the boundary to the fraction s (from 0 to 1)"""
        phi = arctan2(y, x)
        phi = (phi + 2 * pi) % (2 * pi)
        s = self.compute_arc_length(phi) / self.perimeter()
        return s

    def convert_s_to_xy(self, s):
        """Convert s to the point (x, y) on the boundary"""
        arc_length = s * self.perimeter()

        phi = self.arc_length_to_phi(arc_length)

        x, y = self.r_phi(phi) * cos(phi), self.r_phi(phi) * sin(phi)

        return x, y


    def compute_norm_tang(self, x, y):
        """ Compute the outward normal vector and CCW-direction tangent vector on given point (x, y) on the boundary."""
        phi = arctan2(y, x)
        phi = (phi + 2*pi) % (2*pi)
        # Compute dr/dphi using SciPy's derivative
        dr_dphi = derivative(self.r_phi, phi, dx=1e-6)

        # Compute components of the tangent vector
        T_x = dr_dphi * np.cos(phi) - self.r_phi(phi) * np.sin(phi)
        T_y = dr_dphi * np.sin(phi) + self.r_phi(phi) * np.cos(phi)

        # Normalize the tangent vector
        T_norm = np.sqrt(T_x ** 2 + T_y ** 2)
        T_x /= T_norm
        T_y /= T_norm

        # Compute the outward normal vector by rotating the tangent vector 90 degrees
        N_x = T_y
        N_y = -T_x

        norm = np.array([N_x, N_y])
        tang = np.array([T_x, T_y])

        return norm, tang


    def leak(self, n_in, theta, mode='TM'):
        # leak on the boundary, return R for reflection.
        return 1


# Uncomplished definition of stadium-shaped cavity. (convert_s_to_xy(self, s)) 
# class StadiumPolar(BoundaryUniformPolar):
#     def __init__(self, cavity_params):
#         super().__init__(cavity_params)

#     def _set_cavity_params(self, cavity_params):
#         self.R = self.d = cavity_params

#     def __str__(self):
#         return 'StadiumPolar'

#     def r_phi(self, phi):

#         if 1/4 * pi < phi <= 3/4 * pi or 5/4 * pi <= phi <= 7/4 * pi:
#             return np.abs(self.R / np.sin(phi))
#         else:
#             return np.abs(2 * self.R * np.cos(phi))

#     def compute_norm_tang(self, x, y):
#         """
#         :return: normalized normal vector on boundary.
#         """
#         phi = (arctan2(y, x) + 2*pi) % (2 * pi)
#         if 0 <= phi <= 1 / 4 * np.pi:
#             normal_vector = [np.cos(2*phi), np.sin(2*phi)]
#         elif 1 / 4 * np.pi < phi <= 3 / 4 * np.pi:
#             normal_vector = [0, 1]
#         elif 3 / 4 * np.pi < phi < 5 / 4 * np.pi:
#             normal_vector = [-np.cos(2*(phi-pi)), -np.sin(2*(phi-pi))]
#         elif 5 / 4 * np.pi <= phi <= 7 / 4 * np.pi:
#             normal_vector = (0, -1)
#         else:
#             normal_vector = (np.cos(2 * phi), np.sin(2 * phi))

#         return np.array(normal_vector), np.array((-normal_vector[1], normal_vector[0]))

#     def convert_xy_to_s(self, x, y):
#         quarter_perimeter = 1/2 * pi * self.R + self.d
#         perimeter = 4 * quarter_perimeter
#         phi = (arctan2(y, x) + 2*pi) % (2*pi)
#         if phi <= pi/2:
#             if phi <= pi/4:
#                 s = (self.R * 2 * phi) / perimeter
#             else:
#                 s = (quarter_perimeter - x) / perimeter

#         elif pi/2 < phi <= pi:
#             if phi < 3/4 * pi:
#                 s = (quarter_perimeter - x) / perimeter
#             else:
#                 s = (2 * quarter_perimeter - self.R * 2 * (pi - phi)) / perimeter

#         elif pi < phi <= 3*pi/2:
#             if phi < 5/4 * pi:
#                 s = (2 * quarter_perimeter + self.R * 2 * (phi - pi)) / perimeter
#             else:
#                 s = (3 * quarter_perimeter + x) / perimeter

#         else:
#             if phi < 7/4 * pi:
#                 s = (3 * quarter_perimeter + x) / perimeter
#             else:
#                 s = (4 * quarter_perimeter - self.R * 2 * (2*pi - phi)) / perimeter
#         return s

#     def convert_s_to_xy(self, s):
#         if s == 0:
#             return self.R + self.d, 0
#         if s == 3/4:
#             return 0, -self.R


class DshapePolar(BoundaryUniformPolar):
    def __init__(self, cavity_params):
        super().__init__(cavity_params)

    def __str__(self):
        return "D-shapePolar"

    def _set_cavity_params(self, cavity_params):
        self.r, self.d = cavity_params

    def r_phi(self, phi):
        # Range of phi is from 0 to 2pi
        phi_0 = np.arccos(self.d/self.r)
        if phi_0 < phi < (2*pi - phi_0):
            return self.r
        else:
            return self.d / np.cos(phi)

    def compute_norm_tang(self, x, y):
        phi_0 = np.arccos(self.d/self.r)
        phi = (arctan2(y, x) + 2*pi) % (2*pi)
        if phi_0 <= phi <= (2*pi - phi_0):
            normal_vector = np.array([cos(phi), sin(phi)])
        else:
            normal_vector = np.array([1, 0])
        return normal_vector, np.array([-normal_vector[1], normal_vector[0]])

    def convert_xy_to_s(self, x, y):
        phi_0 = np.arccos(self.d/self.r)
        half_line = self.r * sin(phi_0)
        perimeter = 2 * half_line + (2*pi - 2*phi_0) * self.r
        phi = (arctan2(y, x) + 2 * pi) % (2 * pi)

        if phi < phi_0:
            s = self.d * tan(phi) / perimeter
        elif phi_0 <= phi <= (2*pi - phi_0):
            s = (half_line + self.r * (phi - phi_0)) / perimeter
        else:
            s = 1 - (self.d * np.abs(tan(2*pi - phi))) / perimeter

        return s

    def convert_s_to_xy(self, s):
        phi_0 = np.arccos(self.d / self.r)
        half_line = self.r * sin(phi_0)
        perimeter = 2 * half_line + (2 * pi - 2 * phi_0) * self.r
        s_half_line = half_line / perimeter

        if s < s_half_line:
            phi = arctan2(s * perimeter, self.d)
        elif s_half_line <= s <= (1-s_half_line):
            phi = (s * perimeter - half_line) / self.r + phi_0
        else:
            phi = 2*pi - arctan2((1-s) * perimeter, self.d)

        r = self.r_phi(phi)
        return r*cos(phi), r*sin(phi)


class DshapePolarAnalytic(BoundaryUniformPolar):
    def __init__(self, cavity_params):
        super().__init__(cavity_params)

    def __str__(self):
        return "D-shapePolar"

    def _set_cavity_params(self, cavity_params):
        self.r, self.d = cavity_params

    def r_phi(self, phi):
        # Range of phi is from 0 to 2pi
        phi_0 = np.arccos(self.d/self.r)
        if phi_0 < phi < (2*pi - phi_0):
            return self.r
        else:
            return self.d / np.cos(phi)


class Limacon(BoundaryUniformPolar):
    def __init__(self, cavity_params):
        super().__init__(cavity_params)

    def _set_cavity_params(self, cavity_params):
        self.R, self.epsilon = cavity_params

    def r_phi(self, phi):
        return self.R * (1 + 2 * self.epsilon * cos(phi))



def test_is_inside():
    bdryTest = BoundaryUniformPolar((1))
    print(bdryTest.perimeter())
    print(bdryTest.perimeter() - 2*pi)
    plt.plot(bdryTest.bdry_data[:, 0], bdryTest.bdry_data[:, 1])
    for x in np.linspace(0, 2.5, 101):
        for y in np.linspace(-2, 2, 101):
            if bdryTest.is_inside(x, y):
                plt.plot(x, y, '.', c='green')
            else:
                plt.plot(x, y, '*', c='red')
    plt.axis('equal')

    plt.show()


def test_tang_nrom():
    bdryTest = DshapePolarAnalytic((1, 0.5))
    plt.plot(bdryTest.bdry_data[:, 0], bdryTest.bdry_data[:, 1])

    for point in bdryTest.bdry_data[::10]:
        print(point)
        norm, tang = bdryTest.compute_norm_tang(point[0], point[1])
        # plt.plot([point[0], point[0] + norm[0]], [point[1], point[1] + norm[1]])
        plt.plot([point[0], point[0] + tang[0]], [point[1], point[1] + tang[1]])

    plt.axis('scaled')
    plt.show()



if __name__ == '__main__':
    # test_is_inside()
    test_tang_nrom()




