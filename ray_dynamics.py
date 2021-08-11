#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'You Mengyu'

"""
Main program of ray dynamics.
Should be position in the same directory with boundary.py
"""

import numpy as np
import matplotlib.pyplot as plt
import boundary
import time
import os


class RayDynmcs:
    def __init__(self, cavity_type, pars, n_steps=100, initial_condition=(), step_length=0.1):
        """
        :param cavity_type: Class. Represent the geometry of cavity. Should be defined in boundary.py
        :param pars: Turple. Several parameters to define the cavity.
        :param n_steps: Int. Times of reflection of simulation.
        :param initial_condition: Turple. Initial position [turple(phi, sin(theta))] in Brikhoff cordinate.
                                  Default random.
        :param step_length: Defaulted to 0.1, can be adjusted for different size of cavity.

        Example: RayDynmcs('Circle', pars=(10), 1000, initial_condition=(3, 0.2))

        """
        try:
            self.bdry = cavity_type(pars)
        except ValueError or TypeError:
            print('Program can not be excuted.\nCheck the parameters and the type of cavity')
            exit()
        self.n_steps = n_steps
        self.n_steps_left = n_steps
        self.init_condition = initial_condition
        self.step_length = step_length
        self.pars = pars

        self.x, self.y = (0, 0)
        self.pos = np.array([self.x, self.y])
        self.brik = []  # record the point where the reflection happen
        self.pos_info = []  # record the position of ray after every step
        self.motion_vector = np.array([0, 0])

        self.check_initial_condition()
        self.compute_initial_state()

    def check_initial_condition(self):
        """
        Check the validity of input initial_condition, if none, generate random initial condition.
        """
        if self.init_condition == ():
            phi = (np.random.random_sample() - 0.5) * 2 * np.pi
            theta = (np.random.random_sample() / 2) * (np.pi - 0.000001)
            self.brik.append([phi, np.sin(theta)])
            self.init_condition = (phi, np.sin(theta))

        else:
            if not -1 <= self.init_condition[1] <= 1:
                raise ValueError('initial_condition: sin(theta) must between -1 and 1\n')
            phi = self.init_condition[0]
            sin_theta = self.init_condition[1]
            self.brik.append([phi, sin_theta])  # record the initial coordinate in Brikhoff coordination.

    def compute_initial_state(self):
        """
        Compute self.x, self.y, self.brik, and self.motion_vector by inition condition.
        """
        r = self.bdry.bdry(self.init_condition[0])
        phi = self.init_condition[0]
        self.x, self.y = r * np.cos(phi), r * np.sin(phi)
        self.pos = np.array([self.x, self.y])

        self.pos_info.append(self.pos)  # record the inition position

        norm_vector = self.bdry.normal_vector_bdry(self.x, self.y)
        theta = np.arcsin(self.init_condition[1])
        mat_rotation_theta = np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]])
        self.motion_vector = np.matrix.dot(mat_rotation_theta, norm_vector)

        self.pos += self.step_length * self.motion_vector / 100  # Make the ray leave the boundary
        self.pos_info.append(self.pos)

    def test(self):
        print(self.init_condition)
        print(self.x, self.y, self.x**2 + self.y**2)
        print(self.motion_vector)
        print(self.brik)

    def motion(self):
        while self.n_steps_left > 0:
            self.pos = self.pos_info[-1] + self.step_length * self.motion_vector
            print(self.pos)
            x = self.pos[0]
            y = self.pos[1]
            if self.bdry.is_inside(x, y):
                self.pos_info.append(self.pos)
            else:
                corrected_x, corrected_y = self.correct(x, y)
                self.pos = np.array([corrected_x, corrected_y])
                self.pos_info.append(self.pos)
                self.reflect(corrected_x, corrected_y)
                print(f'{self.n_steps_left} / {self.n_steps}')

                self.n_steps_left -= 1

    def correct(self, x, y):
        """
        Because of the length of step, ray may go outside the cavity, so that the correction of position is neccessary.
        """
        while not self.bdry.is_inside(x, y):
            x -= self.motion_vector[0] * self.step_length / 10
            y -= self.motion_vector[1] * self.step_length / 10
        return x, y

    def reflect(self, x, y):
        """
        r = d - 2(d*n)n, r: reflect vector, d: incident vector, n: normal vector
        All of these vectors are normalized. '*' means dot product.
        """
        phi = np.arctan2(y, x)
        norm_vector = self.bdry.normal_vector_bdry(x, y)
        new_motion_vector = self.motion_vector - 2 * (self.motion_vector.dot(norm_vector)) * norm_vector
        self.motion_vector = new_motion_vector
        theta = np.arccos(new_motion_vector.dot(norm_vector))
        self.brik.append([phi, np.sin(theta)])  # save the position when reflection occurs

    def plot_traj(self):
        """Simple plot for test code"""
        bdry_data_ndarray = np.array(self.bdry.bdry_data)
        pos_info_ndarray = np.array(self.pos_info)

        plt.plot(bdry_data_ndarray[:, 0], bdry_data_ndarray[:, 1], c='black', linewidth=1)
        plt.plot(pos_info_ndarray[:, 0], pos_info_ndarray[:, 1], linewidth=0.3, c='black')

        plt.axis('equal')

        plt.show()

    def plot_brik(self):
        """Simple plot for test code"""
        brik_data_ndarray = np.array(self.brik)
        plt.plot(brik_data_ndarray[:, 0], brik_data_ndarray[:, 1], '.', markersize=0.5)
        plt.show()

    def save_data(self):
        """
        Save data in one directory with three different .npy files.
        Process the data by using visualize_data.py
        """
        dirname_0 = f'./{self.bdry}_pars{self.pars}_nsteps{self.n_steps}'
        n = 1
        dirname = dirname_0
        while os.path.exists(dirname):
            dirname = dirname_0 + f'{n}'
            n += 1
        os.mkdir(dirname)

        bdry_data_ndarray = np.array(self.bdry.bdry_data)
        pos_info_ndarray = np.array(self.pos_info)
        brik_data_ndarray = np.array(self.brik)

        np.save(f'{dirname}/boundary_data', bdry_data_ndarray)
        np.save(f'{dirname}/position_data', pos_info_ndarray)
        np.save(f'{dirname}/brikhoff_data', brik_data_ndarray)


def main():
    cavity_type = boundary.Ellipse
    pars = (0.8, 10)  # Check the number of parameters first.
    ray1 = RayDynmcs(cavity_type, pars, n_steps=100)
    ray1.motion()
    ray1.save_data()
    # ray1.plot_traj()
    # ray1.plot_brik()


if __name__ == '__main__':
    main()

