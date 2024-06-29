#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'You Mengyu'

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from numpy import cos, sin
import boundary_uniform


class RayDynamicsUniform:
    def __init__(self, cavity, n_step=100, initial_condition_type="phase", init_condition=(),
                 step_length=1, intensity=1, mode='TM'):
        """
        This class defines a particle-like ray moving in an open billiard.
        :param cavity: Class. The cavity shape defined in boundary.py
        :param n_step: times of reflection of the boundary. Default to 100,
                       means ray simulation will stop at 100th reflection
        :param initial_condition_type: "phase" for phase space, "conf" for configuration space (x-y coordinate).
        :param init_condition: initial condition for the ray.
        :param step_length: length for each step, default to 1e-4.
        :param intensity: Consider the leaking on the boundary, each ray start with intensity=1.
        :param mode: "TM" or "TE", correspond to different leak on the boundary.
        :return: None
        """
        self.motion_vector = None
        self.cavity = cavity
        self.bdry_data = self.cavity.bdry_data
        self.n_step = n_step
        self.n_step_left = n_step

        self.initial_phase_space = None

        self.initial_condition = self._set_initial_condition(initial_condition_type, init_condition)

        self.step_length = step_length
        self.intensity = intensity
        self.mode = mode

        # save [s, p, x, y, intensity] for each reflection (include initial condition).
        self.phase_info = []
        # save [x, y] for each step.
        self.pos_info = []
        # save all the info for each reflection.
        # [s, p, x, y, intensity_reflect, intensity_refract,
        #  reflected_vector, refracted_vector]
        self.phase_info_all = []

        if self.initial_phase_space is not None:
            # The case of initial condition is set to a point in the phase space.
            self.x, self.y = self.cavity.convert_s_to_xy(self.initial_phase_space[0])
            self.pos = np.array([self.x, self.y])
            self.pos_info.append(self.pos)
            self.phase_info.append([self.initial_phase_space[0], self.initial_phase_space[1],
                                    self.x, self.y, self.intensity])
            self._set_initial_motion_vector()

        # self.init_condition = self.init_condition_check(initial_condition_type, init_condition)

    def _set_initial_condition(self, condition_type, condition):
        """Set the initial condition based on the provided type and conditions."""
        if condition_type == "phase":
            if not isinstance(condition, tuple):
                raise ValueError("For 'phase' type, initial condition must be a tuple of two values.")

            if not condition:
                s = np.random.uniform(0 + 1e-10, 1 - 1e-10)
                p = np.random.uniform(-1 + 1e-10, 1 - 1e-10)
            else:
                s, p = condition
                if not (0 <= s < 1) or not (-1 < p < 1):
                    raise ValueError("For 'phase' type initial condition (s, p) must be: 0 <= s < 1 and -1 < p < 1.")

            self.initial_phase_space = (s, p)
            return s, p
        # add elif block if other initial condition type is needed, e.g., (x, y)
        else:
            raise ValueError(f"Unknown initial condition type: {condition_type}")

    def _set_initial_motion_vector(self):
        s, p = self.initial_phase_space
        x, y = self.cavity.convert_s_to_xy(s)
        normal_outward, tang_ccw = self.cavity.compute_norm_tang(x, y)
        normal_inward = -normal_outward
        theta = -np.arcsin(p)

        matrix_rotation_theta = np.array([[cos(theta), -sin(theta)],
                                          [sin(theta), cos(theta)]])

        self.motion_vector = np.matrix.dot(matrix_rotation_theta, normal_inward)

    def is_point_inside_boundary(self, x, y):
        return self.cavity.is_inside(x, y)

    def compute_normal_tangent(self):
        return self.cavity.compute_norm_tang(self.pos[0], self.pos[1])

    def motion(self):
        flag = 0
        while self.n_step_left > 0:
            x, y = self.pos[0], self.pos[1]
            self.pos = self.pos_info[-1] + self.step_length * self.motion_vector

            if self.is_point_inside_boundary(self.pos[0], self.pos[1]):
                # inside the cavity, continue moving
                self.pos_info.append(self.pos)

            else:
                # outside the cavity, come back first
                # corrected_pos = self._correct(self.pos)
                
                # Use bisection method.
                corrected_pos = self._correct_bisect(self.pos)
                # corrected_pos = self._correct(self.pos)
                self.pos = corrected_pos


                # save corrected position
                self.pos_info.append(self.pos)
                self.reflect()
                self.n_step_left -= 1

    def _correct(self, pos):
        while not self.is_point_inside_boundary(pos[0], pos[1]):
            pos -= self.motion_vector * self.step_length / 100
        return pos

    def _correct_bisect(self, pos):
        def func(t):
            x = pos[0]
            y = pos[1]
            vx = self.motion_vector[0]*self.step_length
            vy = self.motion_vector[1]*self.step_length
            x_correct = x - vx*t
            y_correct = y - vy*t
            phi_correct = np.arctan2(y_correct, x_correct)
            phi_correct = (phi_correct + 2*np.pi) % (2*np.pi)
            error = ((x_correct**2 + y_correct**2) - self.cavity.r_phi(phi_correct)**2)
            return error

        t_solution = bisect(func, 0, 0.99999)
        # while func(t_solution) > 0:
        #     t_solution = t_solution + 1e-15

        pos = pos - t_solution * self.motion_vector * self.step_length
        return pos


    def reflect(self):
        s = self.cavity.convert_xy_to_s(self.pos[0], self.pos[1])
        normal_outward, tangent_ccw = self.compute_normal_tangent()


        incident = self.motion_vector
        # compute the reflected vector
        reflected = incident - 2 * incident.dot(normal_outward) * normal_outward
        self.motion_vector = reflected / np.linalg.norm(reflected)

        # p is sin(theta)
        p = np.dot(reflected, tangent_ccw)

        # Compute the angle of incidence using the dot product
        theta_i = np.arccos(-np.dot(incident, normal_outward))

        # Use Snell's law to compute the angle of refraction
        n_in = 3.3
        sin_theta_t = n_in * np.sin(theta_i)

        if sin_theta_t > 1:  # Total internal reflection
            reflectivity = 1
            refracted = np.array([0, 0])

        else:
            # Compute the refracted vector
            # refracted = (n_in * np.cross(normal_outward, np.cross(-normal_outward, incident)) -
            #              normal_outward * np.cos(theta_t))
            refracted = np.array([0, 0])
            reflectivity = self.cavity.leak(n_in, theta_i)
        intensity_reflect = reflectivity * self.intensity
        intensity_refract = (1 - refracted) * self.intensity
        self.intensity = intensity_reflect

        self.phase_info.append([float(s), p, self.pos[0], self.pos[1], self.intensity])
        self.phase_info_all.append([float(s), p, self.pos[0], self.pos[1],
                                    intensity_reflect, intensity_refract,
                                    reflected, refracted])


def test():
    cavity_params = (1, 0.2)
    boundaryTest = boundary.Limacon(cavity_params)

    ray_test = RayDynamicsUniform(boundaryTest, step_length=2, n_step=1000,
                                  init_condition=(7.87864636e-02, 2.94958630e-01))

    time_start = time.time()
    ray_test.motion()
    time_end = time.time()
    print(time_end-time_start)
    phase_info_all = np.array(ray_test.phase_info)
    x = phase_info_all[:, 2]
    y = phase_info_all[:, 3]
    plt.plot(x, y, c='red', linewidth=0.5)
    print(phase_info_all)

    plt.plot(ray_test.bdry_data[:, 0], ray_test.bdry_data[:, 1], 'black', linewidth=1)
    plt.axis('off')
    plt.axis('equal')
    plt.show()
    plt.plot(phase_info_all[:, 0], phase_info_all[:, 1], '.', markersize=2)
    plt.axis([0, 1, -1, 1])
    plt.show()


if __name__ == '__main__':
    test()


