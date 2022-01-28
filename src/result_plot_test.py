import os
from termcolor import cprint
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from utils import *
import pickle
import torch

def from_rpy(roll, pitch, yaw):
    return rotz(yaw).dot(roty(pitch).dot(rotx(roll)))

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                        [s,  c,  0],
                        [0,  0,  1]])

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                        [0,  1,  0],
                        [-s, 0,  c]])

def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                        [0,  c, -s],
                        [0,  s,  c]])

def to_rpy(Rot):
    pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

    if np.isclose(pitch, np.pi / 2.):
        yaw = 0.
        roll = np.arctan2(Rot[0, 1], Rot[1, 1])
    elif np.isclose(pitch, -np.pi / 2.):
        yaw = 0.
        roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
    else:
        sec_pitch = 1. / np.cos(pitch)
        yaw = np.arctan2(Rot[1, 0] * sec_pitch,
                            Rot[0, 0] * sec_pitch)
        roll = np.arctan2(Rot[2, 1] * sec_pitch,
                            Rot[2, 2] * sec_pitch)
    return roll, pitch, yaw

def load(*_file_name):
    pickle_extension = ".p"
    file_name = os.path.join(*_file_name)
    if not file_name.endswith(pickle_extension):
        file_name += pickle_extension
    with open(file_name, "rb") as file_pi:
        pickle_dict = pickle.load(file_pi)
    return pickle_dict

def get_imu_estimates(path_results,dataset_name):
    #  Obtain  estimates
    file_name = os.path.join(path_results+"/", dataset_name + "_imu.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    mondict_imu = load(file_name)
    Rot_imu = mondict_imu['Rot_imu']
    ang_imu = mondict_imu['ang_imu']
    acc_imu = mondict_imu['acc_imu']
    v_imu = mondict_imu['v_imu']
    p_imu = mondict_imu['p_imu']
    b_omega_imu = mondict_imu['b_omega_imu']
    b_acc_imu = mondict_imu['b_acc_imu']
    t_imu = mondict_imu['t_imu']

    return Rot_imu, ang_imu, acc_imu,v_imu, p_imu , b_omega_imu, b_acc_imu, t_imu

def get_wheel_estimates(path_results,dataset_name):
    #  Obtain  estimates
    file_name = os.path.join(path_results+"/", dataset_name + "_wheel.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    mondict_wheel = load(file_name)
    Rot_wheel = mondict_wheel['Rot_wheel']
    ang_wheel = mondict_wheel['ang_wheel']
    v_wheel = mondict_wheel['v_wheel']
    t_wheel = mondict_wheel['t_wheel']
    p_wheel = mondict_wheel['p_wheel']

    return Rot_wheel, ang_wheel, v_wheel , p_wheel, t_wheel

def get_joint_imu_estimates(path_results,dataset_name):
    #  Obtain  estimates
    file_name = os.path.join(path_results+"/", dataset_name + "_joint_imu.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    mondict_wheel = load(file_name)
    Rot_joint_imu = mondict_wheel['Rot_joint_imu']
    ang_joint_imu = mondict_wheel['ang_joint_imu']
    v_joint_imu = mondict_wheel['v_joint_imu']
    t_joint_imu = mondict_wheel['t_joint_imu']
    p_joint_imu = mondict_wheel['p_joint_imu']

    return Rot_joint_imu, ang_joint_imu, v_joint_imu , p_joint_imu, t_joint_imu

def get_gt_data(path_data_save,dataset_name):
    file_name = os.path.join(path_data_save+"/", dataset_name + "_gt.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    mondict_gt = load(file_name)
    
    return mondict_gt['t_gt'], mondict_gt['Rot_gt'], mondict_gt['ang_gt'], mondict_gt['p_gt'], mondict_gt['v_gt']

def results_plot(path_data_save, path_results, dataset_name):
    plt.close('all')
    file_name = os.path.join(path_results+"/", dataset_name + "_imu.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    print("\nResults for: " + dataset_name)

    Rot_imu, ang_imu, acc_imu, v_imu, p_imu, b_omega_imu, b_acc_imu, t_imu = get_imu_estimates(path_data_save,
        dataset_name)

    Rot_wheel, ang_wheel, v_wheel, p_wheel , t_wheel = get_wheel_estimates(path_data_save,dataset_name)

    Rot_joint_imu, ang_joint_imu, v_joint_imu , p_joint_imu, t_joint_imu = get_joint_imu_estimates(path_data_save,dataset_name)

    t_gt, Rot_gt, ang_gt, p_gt, v_gt = get_gt_data(path_data_save,dataset_name)

    v_r_gt = np.zeros((len(v_gt),3))
    v_r_imu = np.zeros((len(v_imu),3))
    v_r_wheel = np.zeros((len(v_wheel),3))

    for j in range(len(Rot_gt)):
        v_r_gt[j] = Rot_gt[j].transpose().dot(v_gt[j])
    for j in range(len(Rot_imu)):
        v_r_imu[j] = Rot_imu[j].transpose().dot(v_imu[j])
    for j in range(len(Rot_wheel)):
        v_r_wheel[j] = Rot_wheel[j].transpose().dot(v_wheel[j])

    # plot and save plot
    folder_path = os.path.join(path_results, dataset_name)
    create_folder(folder_path)

    # rot_error = ang_gt[-1,2] - ang_imu[-2,2]
    # print(dataset_name+" rot gt: " ,ang_gt[-1,2], " rot imu: ", ang_imu[-2,2])
    # print(dataset_name+" rot error: " ,rot_error)

    # p_error = p_gt[-1,:2] - p_wheel[-1,:2]
    # print(dataset_name+" p gt: " ,p_gt[-1,:2], " p wheel: ", p_wheel[-1,:2])
    # print(dataset_name+" p error: " ,p_error)

    # ax1 = plt.figure().add_subplot(figsize=(20, 10))
    # x, y, z = np.meshgrid(p_gt[:,0],
    #                   p_gt[:,1])

    # u = v_gt[:,0]
    # v = v_gt[:,1]

    # ax1.quiver(x, y, u, v, length=1, normalize=True)

    # plt.show()

    # velocity
    fig1, ax1 = plt.subplots(sharex=True, figsize=(20, 10))

    ax1.plot(t_gt, v_gt[:,:2])
    ax1.plot(t_joint_imu, v_joint_imu[:,:2])
    # ax1.plot(t_imu, v_imu[:,:2])
    # ax1.plot(t_wheel, v_wheel[:,:2])

    ax1.set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
    
    ax1.grid()


    ax1.legend(
        ['$gt^x$', '$gt^y$', '$joint imu^x$', '$joint imu^y$'])
        # ['$gt^x$', '$gt^y$', '$imu^x$', '$imu^y$'])
        # ['$mocap^x$', '$mocap^y$', '$imu^x$', '$imu^y$', '$wheel^x$', '$wheel^y$'])


    # # position, velocity and velocity in body frame
    # fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

    # axs1[0].plot(t_gt, p_gt[:,:2])
    # axs1[0].plot(t_imu, p_imu[:,:2])
    # axs1[0].plot(t_wheel, p_wheel[:,:2])
    # axs1[1].plot(t_gt, v_gt[:,:2])
    # axs1[1].plot(t_imu, v_imu[:,:2])
    # axs1[1].plot(t_wheel, v_wheel[:,:2])
    # axs1[2].plot(t_gt, v_r_gt[:,:2])
    # axs1[2].plot(t_imu, v_r_imu[:,:2])
    # axs1[2].plot(t_wheel, v_r_wheel[:,:2])

    # axs1[0].set(xlabel='time (s)', ylabel='$\mathbf{p}_n$ (m)', title="Position")
    # axs1[1].set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
    # axs1[2].set(xlabel='time (s)', ylabel='$\mathbf{R}_n^T \mathbf{v}_n$ (m/s)',
    #             title="Velocity in body frame")  

    # for ax in chain(axs1):
    #     ax.grid()

    # axs1[0].legend(
    #     ['$mocap^x$', '$mocap^y$', '$imu^x$', '$imu^y$', '$wheel^x$', '$wheel^y$'])
    #     # ['$mocap^x$', '$mocap^y$', '$mocap^z$', '$imu^x$', '$imu^y$', '$imu^z$', '$wheel^x$', '$wheel^y$', '$wheel^z$'])
    #     # ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
    # axs1[1].legend(
    #     ['$mocap^x$', '$mocap^y$', '$imu^x$', '$imu^y$', '$wheel^x$', '$wheel^y$'])
    #     # ['$mocap^x$', '$mocap^y$', '$mocap^z$', '$imu^x$', '$imu^y$', '$imu^z$', '$wheel^x$', '$wheel^y$', '$wheel^z$'])
    #     # ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
    # axs1[2].legend(
    #     ['$mocap^x$', '$mocap^y$', '$imu^x$', '$imu^y$', '$wheel^x$', '$wheel^y$'])
    #     # ['$mocap^x$', '$mocap^y$', '$mocap^z$', '$imu^x$', '$imu^y$', '$imu^z$', '$wheel^x$', '$wheel^y$', '$wheel^z$'])
    #     # ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])


    # orientation, bias gyro and bias accelerometer
    fig2, ax2 = plt.subplots(sharex=True, figsize=(20, 10))
    
    pi_up = np.zeros(len(t_gt))
    pi_down = np.zeros(len(t_gt))
    for j in range(len(Rot_gt)):
        pi_up[j] = 3.141592
        pi_down[j] = -3.141592

    # ax2.plot(t_gt, ang_gt[:,2])
    # ax2.plot(t_imu, ang_imu[:,2])
    ax2.plot(t_gt, ang_gt[:,2])
    ax2.plot(t_imu, ang_imu[:,2])
    # ax2.plot(t_wheel, ang_wheel[:,2])
    ax2.plot(t_gt, pi_up, 'k',linestyle='--')
    ax2.plot(t_gt, pi_down, 'k',linestyle='--')

    ax2.set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
                title="Orientation")

    ax2.grid()

    # ax2.legend([r'gt', r'wheel'])
    ax2.legend([r'gt', r'imu'])
    # ax2.legend([r'gt', r'imu', r'wheel'])



    # position in plan
    fig3, ax3 = plt.subplots(figsize=(20, 10))

    ax3.plot(p_gt[:, 0], p_gt[:, 1])
    ax3.plot(p_joint_imu[:, 0], p_joint_imu[:, 1])
    # ax3.plot(p_imu[:, 0], p_imu[:, 1])
    # ax3.plot(p_wheel[:, 0], p_wheel[:, 1])
    ax3.axis('equal')

    ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")

    ax3.grid()

    ax3.legend(['gt', 'joint_imu'])
    # ax3.legend(['gt', 'wheel'])
    # ax3.legend(['gt', 'imu', 'wheel'])
    # ax3.legend(['gt'])


    # # acc in plan
    # fig4, ax4 = plt.subplots(figsize=(20, 10))
    
    # ax4.plot(t_imu, acc_imu)

    # ax4.grid()

    # ax4.legend(
    #     ['$a_n^x$', '$a_n^y$', '$a_n^z$'])    
        
    # # save figures
    # figs = [fig1, fig2, fig3]
    # figs_name = ["position_velocity", "orientation", "position_xy"]
    # # figs = [fig3]
    # # figs_name = ["position_xy"]
    # for l, fig in enumerate(figs):
    #     fig_name = figs_name[l]
    #     fig.savefig(os.path.join(folder_path, fig_name + ".png"))


    plt.show(block=True)

