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

    return Rot_imu, ang_imu, acc_imu, v_imu, p_imu , b_omega_imu, b_acc_imu, t_imu

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

def get_joint_estimates(path_results,dataset_name):
    #  Obtain  estimates
    file_name = os.path.join(path_results+"/", dataset_name + "_joint.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    mondict_joint = load(file_name)
    Rot_joint = mondict_joint['Rot_joint']
    ang_joint = mondict_joint['ang_joint']
    v_joint = mondict_joint['v_joint']
    t_joint = mondict_joint['t_joint']
    p_joint = mondict_joint['p_joint']

    return Rot_joint, ang_joint, v_joint , p_joint, t_joint

def get_gt_data(path_data_save,dataset_name):
    file_name = os.path.join(path_data_save+"/", dataset_name + "_gt.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    mondict_gt = load(file_name)
    
    return mondict_gt['t_gt'], mondict_gt['Rot_gt'], mondict_gt['ang_gt'], mondict_gt['p_gt'], mondict_gt['v_gt']

def results_plot_lab(path_data_save, path_results, dataset_name):
    plt.close('all')
    file_name = os.path.join(path_results+"/", dataset_name + "_imu.p")
    if not os.path.exists(file_name):
        print('No result for ' + dataset_name)
        return

    print("\nResults for: " + dataset_name)

    Rot_imu, ang_imu, acc_imu, v_imu, p_imu, b_omega_imu, b_acc_imu, t_imu = get_imu_estimates(path_results,
        dataset_name)

    Rot_wheel, ang_wheel, v_wheel, p_wheel , t_wheel = get_wheel_estimates(path_results,dataset_name)
    Rot_joint, ang_joint, v_joint, p_joint, t_joint = get_joint_estimates(path_results,dataset_name)
    # t_gt, Rot_gt, ang_gt, p_gt, v_gt = get_gt_data(path_data_save,dataset_name)

    # v_r_gt = np.zeros((len(v_gt),3))
    v_r_imu = np.zeros((len(v_imu),3))
    v_r_wheel = np.zeros((len(v_wheel),3))

    # for j in range(len(Rot_gt)):
    #     v_r_gt[j] = Rot_gt[j].transpose().dot(v_gt[j])
    for j in range(len(Rot_imu)):
        v_r_imu[j] = Rot_imu[j].transpose().dot(v_imu[j])
    for j in range(len(Rot_wheel)):
        v_r_wheel[j] = Rot_wheel[j].transpose().dot(v_wheel[j])

    # plot and save plot
    folder_path = os.path.join(path_results, dataset_name)
    create_folder(folder_path)

    # ax1 = plt.figure().add_subplot(figsize=(20, 10))
    # x, y, z = np.meshgrid(p_gt[:,0],
    #                   p_gt[:,1])

    # u = v_gt[:,0]
    # v = v_gt[:,1]

    # ax1.quiver(x, y, u, v, length=1, normalize=True)

    # plt.show()

    # # position, velocity and velocity in body frame
    fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
    # orientation, bias gyro and bias accelerometer
    fig2, ax2 = plt.subplots(sharex=True, figsize=(20, 10))
    # position in plan
    fig3, ax3 = plt.subplots(figsize=(20, 10))
    # acc in plan
    fig4, ax4 = plt.subplots(figsize=(20, 10))

    axs1[0].plot(t_imu, p_imu)
    axs1[0].plot(t_wheel, p_wheel)
    # axs1[0].plot(t_joint, p_joint)

    axs1[1].plot(t_imu, v_imu)
    axs1[1].plot(t_wheel, v_wheel)
    # axs1[1].plot(t_joint, v_joint)

    axs1[2].plot(t_imu, v_r_imu)
    axs1[2].plot(t_wheel, v_r_wheel)

    pi_up = np.zeros(len(t_imu))
    pi_down = np.zeros(len(t_imu))
    for j in range(len(Rot_imu)):
        pi_up[j] = 3.141592
        pi_down[j] = -3.141592

    ax2.plot(t_imu, ang_imu)
    ax2.plot(t_wheel, ang_wheel)
    ax2.plot(t_imu, pi_up, 'k',linestyle='--')
    ax2.plot(t_imu, pi_down, 'k',linestyle='--')


    ax3.plot(p_imu[:, 0], p_imu[:, 1])
    ax3.plot(p_wheel[:, 0], p_wheel[:, 1])
    # ax3.plot(p_joint[:, 0], p_joint[:, 1])
    ax3.axis('equal')

    ax4.plot(t_imu, acc_imu)


    axs1[0].set(xlabel='time (s)', ylabel='$\mathbf{p}_n$ (m)', title="Position")
    axs1[1].set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
    axs1[2].set(xlabel='time (s)', ylabel='$\mathbf{R}_n^T \mathbf{v}_n$ (m/s)',
                title="Velocity in body frame")
    ax2.set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
                title="Orientation")
    ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")

    for ax in chain(axs1):
        ax.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    axs1[0].legend(
        # ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$', '$\hat{\hat{p}}_n^x$' '$\hat{\hat{p}}_n^y$', '$\hat{\hat{p}}_n^z$'])
        ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
    axs1[1].legend(
        # ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$', '$\hat{\hat{v}}_n^x$', '$\hat{\hat{v}}_n^y$', '$\hat{\hat{v}}_n^z$'])
        ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
    axs1[2].legend(
        # ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$', '$\hat{\hat{v}}_n^x$', '$\hat{\hat{v}}_n^y$', '$\hat{\hat{v}}_n^z$'])
        ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
    # ax2.legend([r'$\phi_n^x$', r'$\theta_n^y$', r'$\psi_n^z$', r'$\hat{\phi}_n^x$',
    #                 r'$\hat{\theta}_n^y$', r'$\hat{\psi}_n^z$', r'$\hat{\hat{\phi}}_n^x$',
    #                 r'$\hat{\hat{\theta}}_n^y$', r'$\hat{\hat{\psi}}_n^z$'])
    ax2.legend([r'$\phi_n^x$', r'$\theta_n^y$', r'$\psi_n^z$', r'$\hat{\phi}_n^x$',
                    r'$\hat{\theta}_n^y$', r'$\hat{\psi}_n^z$'])
    # ax2.legend([r'$gt^z$', r'$imu^z$', r'$wheel^z$'])
    # ax3.legend(['ground-truth trajectory', 'imu', 'wheel'])
    # ax3.legend(['ground-truth trajectory', 'wheel'])
    ax3.legend(['imu', 'wheel'])
    # ax3.legend(['wheel', 'joint'])

    ax4.legend(
        ['$p_n^x$', '$p_n^y$', '$p_n^z$'])    

    # save figures
    figs = [fig1, fig2, fig3]
    # figs = [fig2,fig3]
    figs_name = ["position_velocity", "orientation", "position_xy"]
    # figs_name = ["orientation", "position_xy"]
    for l, fig in enumerate(figs):
        fig_name = figs_name[l]
        fig.savefig(os.path.join(folder_path, fig_name + ".png"))

    plt.show(block=True)