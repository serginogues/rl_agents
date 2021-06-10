"""
https://pypi.org/project/tinyik/
"""
import tinyik
import numpy as np
# two joints that rotate around z-axis and two links of 100.0 length along x-axis
arm = tinyik.Actuator(['z', [100., 0., 0.], 'z', [100., 0., 0.]])


def angle_to_ee(angles_=np.deg2rad([30, 60])):
    arm.angles = angles_
    return arm.ee


def ee_to_angle(ee_):
    arm.ee = [ee_[0], ee_[1], 0]
    return arm.angles  # np.round(np.rad2deg(arm.angles))


def twoD_robot_arm_goal(target_pos):
    # calculate ideal joint positions
    angle = ee_to_angle(target_pos)
    goal = [target_pos[0], target_pos[1], angle[0], angle[1]]
    return goal