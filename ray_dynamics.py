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
import os
from datetime import datetime


class RayDynmcs:
    def __init__(self, cavity_type, pars, n_steps=100,
                 initial_condition=(), step_length=0.01, intensity=1, with_tailor=0):
        """
        :param cavity_type: Class. Represent the geometry of cavity. Should be defined in boundary.py
        :param pars: Turple. Several parameters to define the cavity.
        :param n_steps: Int. Times of reflection of simulation.
        :param initial_condition: Turple. Initial position [turple(s, sin(theta))] in Brikhoff cordinate.
                                  Default random.
        :param step_length: Defaulted to 0.01, can be adjusted for different size of cavity.
        :param with_tailor: 0(default) means without tailor inside cavity

        Example: RayDynmcs('Circle', pars=(10), 1000, initial_condition=(0.1, 0.2))

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
        self.intensity = intensity

        self.x, self.y = (0, 0)
        self.pos = np.array([self.x, self.y])
        self.brik_init = []
        self.brik = []  # record the point where the reflection happen
        self.pos_info = []  # record the position of ray after every step
        self.pos_info_plot = []  # record the position of ray at te reflection happens
        self.full_info = []  # record all the information during the motion
        self.motion_vector = np.array([0, 0])
        self.with_tailor = with_tailor
        if self.with_tailor:
            self.tailored_phase_info = []
            self.is_tailored = 0
        self.check_initial_condition()
        self.compute_initial_state()

    def check_initial_condition(self):
        """
        Check the validity of input initial_condition, if none, generate random initial condition.
        """
        if self.init_condition == ():
            s = np.random.random_sample()
            theta = (np.random.random_sample() / 2) * (np.pi - 0.000001)
            self.brik.append([s, np.sin(theta), self.intensity])
            self.init_condition = (s, np.sin(theta))

        else:
            if not -1 <= self.init_condition[1] <= 1:
                raise ValueError('initial_condition: sin(theta) must between -1 and 1\n')
            s = self.init_condition[0]
            sin_theta = self.init_condition[1]
            self.brik_init = [s, sin_theta, self.intensity]  # record the initial coordinate.

    def compute_initial_state(self):
        """
        Compute self.x, self.y, self.brik, and self.motion_vector by inition condition.
        """
        r = self.bdry.bdry(self.bdry.convert_s_phi(self.init_condition[0]))
        phi = self.bdry.convert_s_phi(self.init_condition[0])
        self.x, self.y = r * np.cos(phi), r * np.sin(phi)
        self.pos = np.array([self.x, self.y])
        self.brik.append([self.brik_init[0], self.brik_init[1], self.intensity, phi, self.x, self.y])

        self.pos_info.append(self.pos)  # record the inition position
        self.full_info.append([self.x, self.y, self.intensity])
        self.pos_info_plot.append(self.pos)

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
            x = self.pos[0]
            y = self.pos[1]
            if self.with_tailor:
                if self.bdry.is_tailor(x, y):
                    self.n_steps_left = 1
                    self.is_tailored = 1
            if self.bdry.is_inside(x, y):
                self.pos_info.append(self.pos)
                self.full_info.append([self.pos_info[-1][0], self.pos_info[-1][1], self.intensity])
            else:
                corrected_x, corrected_y = self.correct(x, y)
                self.pos = np.array([corrected_x, corrected_y])
                self.pos_info.append(self.pos)
                self.full_info.append([self.pos_info[-1][0], self.pos_info[-1][1], self.intensity])
                self.pos_info_plot.append(self.pos)
                self.reflect(corrected_x, corrected_y)

                # print(f'{self.n_steps_left} / {self.n_steps}')

                self.n_steps_left -= 1
                if self.intensity == 0:
                    self.n_steps_left = 0

    def correct(self, x, y):
        """
        Because of the length of step, ray may go outside the cavity, so that the correction of position is neccessary.
        """
        while not self.bdry.is_inside(x, y):
            x -= self.motion_vector[0] * self.step_length / 1000
            y -= self.motion_vector[1] * self.step_length / 1000
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

        self.intensity = self.bdry.leak(phi, theta) * self.intensity  # leak

        s = self.bdry.convert_phi_s(phi)

        if self.motion_vector.dot(self.bdry.boundary_vector(s)) < 0:
            flag = 1  # clockwise
        else:
            flag = -1  # counterclockwise

        self.brik.append([s, flag*np.sin(theta), self.intensity, phi, x, y])  # save all info when reflection occurs
        if self.with_tailor:
            if self.is_tailored:

                self.tailored_phase_info = self.brik

        # self.brik.append([self.bdry.convert_phi_s(phi), np.sin(theta)])  # save the position when reflection occurs

    def plot_traj(self):
        """Simple plot for test code"""
        bdry_data_ndarray = np.array(self.bdry.bdry_data)
        pos_info_ndarray = np.array(self.pos_info_plot)

        plt.figure(figsize=(4, 4))
        plt.plot(bdry_data_ndarray[:, 0], bdry_data_ndarray[:, 1], c='black')
        plt.plot(pos_info_ndarray[:, 0], pos_info_ndarray[:, 1], linewidth=0.5, c='black')
        plt.axis('off')
        plt.axis('equal')


        plt.show()

    def plot_brik_traj(self):
        """Simple plot for test code"""
        brik_data_ndarray = np.array(self.brik)
        print(brik_data_ndarray[0][0], brik_data_ndarray[1][0])
        plt.plot(brik_data_ndarray[:, 0], brik_data_ndarray[:, 1], '.', markersize=10, c='black')
        plt.xlim([0, 1])
        plt.ylim([-1, 1])

        plt.show()

    def save_data(self):
        """
        Save data in one directory with three different .npy files.
        Process the data by using visualize_data.py
        """
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        dirname = f'./{self.bdry}_pars{self.pars}_nsteps{self.n_steps}_{time_now}'
        os.mkdir(dirname)

        bdry_data_ndarray = np.array(self.bdry.bdry_data)
        pos_info_ndarray = np.array(self.pos_info)
        brik_data_ndarray = np.array(self.brik)

        np.save(f'{dirname}/boundary_data', bdry_data_ndarray)
        np.save(f'{dirname}/position_data', pos_info_ndarray)
        np.save(f'{dirname}/brikhoff_data', brik_data_ndarray)

    def standard(self):
        self.motion()
        self.save_data()


def random_init_full():
    s = np.random.random()
    sin_theta = np.random.uniform(-1, 1)
    return s, sin_theta


def random_init_specific_circle(rad, center=(0, 0)):
    x0, y0 = center
    t = np.random.uniform(0.0, 2.0 * np.pi)
    r = rad * np.sqrt(np.random.uniform(0.0, 1.0))
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)

    return x, y


def full_brik(cavity_type, pars):
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    dirname = f'./{cavity_type(pars)}_full_brik_data_{time_now}'


    os.mkdir(dirname)

    flag = 1
    for _ in range(200000):
        initial_condition = random_init_full()
        ray = RayDynmcs(cavity_type, pars, n_steps=30, initial_condition=initial_condition)
        ray.motion()
        brik_data_ndarray = np.array(ray.brik)
        # full_info = np.array(ray.full_info)
        # print(brik_data_ndarray)
        fname_brik = f's{initial_condition[0]}_{initial_condition[1]}_brik'
        # fname_full = f's{initial_condition[0]}_{initial_condition[1]}_full'
        if os.path.exists(f'./{dirname}/{fname_brik}.npy'):
            continue
        np.save(f'./{dirname}/{fname_brik}', brik_data_ndarray)
        # np.save(f'./{dirname}/{fname_full}', full_info)

        flag += 1
        # print(flag)


def brik_with_tailor(cavity_type=boundary.D, pars=(1, 2)):
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    dirname = f'./{cavity_type(pars)}_tailored_brik_info_{time_now}'

    os.mkdir(dirname)

    flag = 0
    brik_tailored = []
    num = int(10e5)
    for _ in range(num):
        initial_condition = random_init_full()
        ray = RayDynmcs(cavity_type, pars, n_steps=1, initial_condition=initial_condition, with_tailor=1)
        ray.motion()
        if ray.is_tailored:
            # print('SUCKED')
            brik_tailored.append(ray.tailored_phase_info)

        flag += 1
        # print(f'{flag} / {num} are finished')

    fname_brik = f'tailored_brik_info_{num}init_pts'
    data_save = np.array(brik_tailored)
    np.save(f'./{dirname}/{fname_brik}', data_save)


def brik_upo(cavity_type, pars):
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    num = 200000
    center = (0.29798, 0.36568)
    rad = 0.02

    dirname = f'./{cavity_type(pars)}_brik_center{center}_rad{rad}_{time_now}'
    os.mkdir(dirname)

    flag = 1
    for _ in range(num):
        initial_condition = random_init_specific_circle(rad, center=center)
        ray = RayDynmcs(cavity_type, pars, n_steps=40, initial_condition=initial_condition)
        ray.motion()
        brik_data_ndarray = np.array(ray.brik)
        # full_info = np.array(ray.full_info)
        # print(brik_data_ndarray)
        fname_brik = f's{initial_condition[0]}_{initial_condition[1]}_brik'
        # fname_full = f's{initial_condition[0]}_{initial_condition[1]}_full'
        if os.path.exists(f'./{dirname}/{fname_brik}.npy'):
            continue
        np.save(f'./{dirname}/{fname_brik}', brik_data_ndarray)
        # np.save(f'./{dirname}/{fname_full}', full_info)

        flag += 1
        # print(flag)


def test():
    cavity_type = boundary.D
    pars = (1, 2)  # Check the number of parameters first.
    ray = RayDynmcs(cavity_type, pars, initial_condition=(2.980488693183152571e-01, -3.659702357060121702e-01), n_steps=3)

    ray.motion()
    # ray1.save_data()
    ray.plot_traj()
    print(ray.pos_info)
    # ray.plot_brik_traj()
    # np.savetxt('UPO_D/p7_points', ray.brik)
    # np.savetxt('UPO_D/p7_traj', ray.pos_info)
    # ray1.save_data()
    # brik_data_ndarray = np.array(ray1.brik)
    # plt.plot((brik_data_ndarray[:, 0][::2]), c='black', linewidth=0.5)
    # plt.show()
    # full_brik(cavity_type, pars)


if __name__ == '__main__':
    # test()
    full_brik(boundary.D, (1, 2))
    # print(random_init_full())
    # random_init_specific_circle()
    # brik_upo(boundary.D, (1, 2))
    # brik_with_tailor()

# test()

