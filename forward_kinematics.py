import numpy as np
import visual_kinematics

"""
https://pypi.org/project/visual-kinematics/
https://github.com/yongan007/inverse_foward_kinematics_Kuka_iiwa/blob/master/homework_01_report.pdf

# joint limits (not needed here)
phi1_max = np.deg2rad(170)
phi2_max = np.deg2rad(120)
phi3_max = np.deg2rad(170)
phi4_max = np.deg2rad(120)
phi5_max = np.deg2rad(170)
phi6_max = np.deg2rad(120)
phi7_max = np.deg2rad(175)

- dh_params = n*4 matrix where n = number of joints
- columns = d, theta, a, alpha:
        - d:        link offset (constant)
        - a:        link length (constant)
        - alpha:    link twist (constant)
        - theta:    joint angle
        
- joint_i = [d_i, theta_i, a_i, alpha_i]
"""
# LBR iiwa 7 R800 DH parameters:
# d (mm)
d_bs = .340
d_se = .400
d_ew = .400
d_wf = .126

# a
a = 0

# alpha
alpha_1 = -np.pi / 2
alpha_2 = np.pi / 2
alpha_3 = -np.pi / 2
alpha_4 = np.pi / 2
alpha_5 = -np.pi / 2
alpha_6 = np.pi / 2
alpha_7 = 0

# joint_i = [d_i, theta_i, a_i, alpha_i]
joint_1 = [d_bs, 0, a, alpha_1]
joint_2 = [0, 0, a, alpha_2]
joint_3 = [d_se, 0, a, alpha_3]
joint_4 = [0, 0, a, alpha_4]
joint_5 = [d_ew, 0, a, alpha_5]
joint_6 = [0, 0, a, alpha_6]
joint_7 = [d_wf, 0, a, alpha_7]

dh_params = np.array([joint_1,
                      joint_2,
                      joint_3,
                      joint_4,
                      joint_5,
                      joint_6,
                      joint_7])


"""# DH parameters of Aubo-i10
dh_params = np.array([[0.163, 0., 0., 0.5 * np.pi],
                      [0., 0.5 * np.pi, 0.632, np.pi],
                      [0., 0., 0.6005, np.pi],
                      [0.2013, -0.5 * np.pi, 0., -0.5 * np.pi],
                      [0.1025, 0., 0., 0.5 * np.pi],
                      [0.094, 0., 0., 0.]])"""

robot = visual_kinematics.Robot(dh_params)


def forward(theta):
    """
    You can also get the end frame by calling the Robot's property end_frame
    robot.end_frame
    :param theta:
    :return:
    """
    f = robot.forward(theta)
    print("-------forward-------")
    print("end frame t_4_4:")
    print(f.t_4_4)
    print("end frame xyz:")
    print(f.t_3_1.reshape([3, ]))
    print("end frame abc:")
    print(f.euler_3)
    print("end frame rotational matrix:")
    print(f.r_3_3)
    print("end frame quaternion:")
    print(f.q_4)
    print("end frame angle-axis:")
    print(f.r_3)
    return f

print(dh_params)
theta = np.array([0., 0, 0, 0., 0., 0., 0.])
print(theta)
"""x: -0.06025666335711789
y: -0.06830510125898295
z: 1.197979162853224"""
forward(theta)


# get batch vectors: