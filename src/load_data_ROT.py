import torch
import numpy as np
import pandas as pd
import os
import math
import pickle
import copy
from result_plot_ROT import *
from scipy.signal import filtfilt, butter
from scipy.spatial.transform import Rotation as R
from lie_algebra import SO3

def skew(x):
    X = np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])
    return X

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

def mat_exp(omega):
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    angle = np.linalg.norm(omega)

    # Near phi==0, use first order Taylor expansion
    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)

def so3exp(phi):
    angle = np.linalg.norm(phi)

    # Near phi==0, use first order Taylor expansion
    if np.abs(angle) < 1e-8:
        skew_phi = np.array([[0, -phi[2], phi[1]],
                    [phi[2], 0, -phi[0]],
                    [-phi[1], phi[0], 0]])
        return np.identity(3) + skew_phi

    axis = phi / angle
    skew_axis = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * skew_axis

def normalize_rot(Rot):

    # The SVD is commonly written as a = U S V.H.
    # The v returned by this function is V.H and u = U.
    U, _, V = np.linalg.svd(Rot, full_matrices=False)

    S = np.identity(3)
    S[2, 2] = np.linalg.det(U) * np.linalg.det(V)
    return U.dot(S).dot(V)

def run_imu(t_imu,Rot_imu, acc_imu):
    dt_imu = t_imu[1:] - t_imu[:-1]
    N = len(acc_imu)

    v_imu = np.zeros((N, 3))
    p_imu = np.zeros((N, 3))
    b_omega_imu = np.zeros((N, 3))
    b_acc_imu = np.zeros((N, 3))

    for i in range(1,N):
        v_imu[i], p_imu[i], b_omega_imu[i], b_acc_imu[i] = propagate_imu(Rot_imu[i-1], v_imu[i-1], p_imu[i-1], b_omega_imu[i-1], b_acc_imu[i-1], acc_imu[i], dt_imu[i-1])

    return v_imu, p_imu, b_omega_imu, b_acc_imu


def propagate_imu(Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, acc_prev, dt):
    acc = Rot_prev.dot(acc_prev - b_acc_prev)
    v = v_prev + acc * dt
    p = p_prev + v_prev*dt + 1/2 * acc * dt**2
    b_omega = b_omega_prev
    b_acc = b_acc_prev

    return v, p, b_omega, b_acc

def run_joint(t_joint,v_wheel_joint,Rot_joint):
    dt_joint = t_joint[1:] - t_joint[:-1]
    N = len(v_wheel_joint)
    p_joint = np.zeros((N, 3))
    v_joint = np.zeros((N, 3))
    # acc_joint = np.zeros((N, 3))

    for i in range(1,N):
        # acc_joint[i] = (v_wheel_joint[i]-v_wheel_joint[i-1])/dt_joint[i-1]
        p_joint[i], v_joint[i] = propagate_joint(Rot_joint[i-1], p_joint[i-1], v_wheel_joint[i-1], dt_joint[i-1])

    return p_joint, v_joint

def propagate_joint(Rot_prev, p_prev, v_prev, dt):
    # acc = Rot_prev.dot(a)
    # v = Rot_prev.dot(v_prev) + acc*dt
    # p = p_prev + v*dt + 0.5*acc*(dt**2)
    v = Rot_prev.dot(v_prev)
    p = p_prev + v*dt
    return p, v

def dump(mondict, *_file_name):
    pickle_extension = ".p"
    file_name = os.path.join(*_file_name)
    if not file_name.endswith(pickle_extension):
        file_name += pickle_extension
    with open(file_name, "wb") as file_pi:
        pickle.dump(mondict, file_pi)

def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q[x,y,z,w]
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 1 - 2 * (q1 * q1 + q2 * q2)
    r01 = 2 * (q0 * q1 - q2 * q3)
    r02 = 2 * (q0 * q2 + q1 * q3)
     
    # Second row of the rotation matrix
    r10 = 2 * (q0 * q1 + q2 * q3)
    r11 = 1 - 2 * (q0 * q0 + q2 * q2)
    r12 = 2 * (q1 * q2 - q0 * q3)
     
    # Third row of the rotation matrix
    r20 = 2 * (q0 * q2 - q1 * q3)
    r21 = 2 * (q1 * q2 + q0 * q3)
    r22 = 1 - 2 * (q0 * q0 + q1 * q1)
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
    
def direction_vector_to_rotation_matrix(V,g):
    v = np.zeros((3,))
    for i in range(50):
        v = v + V[i]
    v = v/100
    r = np.linalg.norm(v)
    v = v/r
    g = g * r
    R = np.identity(3)
    R[:,0] = R[:,0].dot(v)
    R[:,1] = R[:,1].dot(v)
    R[:,2] = R[:,2].dot(v)

    return R.transpose().dot(g)

def high_pass_filter(acc_imu, cut_value):
    N = len(acc_imu)
    a = np.zeros((N,3))
    # dt = np.diff(t_imu)
    # filt_cut_off = 0.01

    # a = filtfilt(*butter(1,filt_cut_off,btype='lowpass'),acc_imu,axis=0)
    
    for i in range(N):
        if ((abs(acc_imu[i,1]) < cut_value[0]) or (abs(acc_imu[i,1]) > cut_value[1])):
            a[i,1] = 0
        else : a[i,1] = acc_imu[i,1]
    
    return a

def interp_data(t_x, x, t):
    x_int = np.zeros((t.shape[0], x.shape[1]))
    for i in range(0, x.shape[1]):
            x_int[:, i] = np.interp(t, t_x, x[:, i])
    return x_int

if __name__ == '__main__':
    datasets = ["square_ccw","square_cw","circle_ccw","circle_cw","ribbon","inf","random_1","random_2"]
    # datasets = ['random_2']
    for dataset_name in datasets:
        path_data_base = "../../../Datasets/husky_dataset/211207/"+dataset_name +"/"
        path_data_save = "../data"
        path_results = "../results"

        date_dirs = os.listdir(path_data_base)
        for n_iter, date_dir in enumerate(date_dirs):
            if (date_dir == dataset_name+"_mocap.txt"):
                path1 = os.path.join(path_data_base, date_dir)
                gt_csv = pd.read_csv(path1,sep="\t")
                gt_data = gt_csv.to_numpy()

                gt_front_left = torch.Tensor(gt_data[:,1:4]/1e3).double()
                gt_front_right = torch.Tensor(gt_data[:,4:7]/1e3).double()
                gt_back_left = torch.Tensor(gt_data[:,7:10]/1e3).double()
                gt_back_right = torch.Tensor(gt_data[:,10:13]/1e3).double()

                y_axis_temp = ((gt_front_left - gt_front_right) + (gt_back_left - gt_back_right))/2
                x_axis_temp = ((gt_front_right - gt_back_right) + (gt_front_left - gt_back_left))/2
                z_axis_temp = bmv(SO3.wedge(x_axis_temp),y_axis_temp)

                x_length = torch.sqrt(torch.pow(x_axis_temp[:,0],2) + torch.pow(x_axis_temp[:,1],2) + torch.pow(x_axis_temp[:,2],2))
                y_length = torch.sqrt(torch.pow(y_axis_temp[:,0],2) + torch.pow(y_axis_temp[:,1],2) + torch.pow(y_axis_temp[:,2],2))
                z_length = torch.sqrt(torch.pow(z_axis_temp[:,0],2) + torch.pow(z_axis_temp[:,1],2) + torch.pow(z_axis_temp[:,2],2))

                x_axis = torch.zeros_like(x_axis_temp).double()
                y_axis = torch.zeros_like(y_axis_temp).double()
                z_axis = torch.zeros_like(z_axis_temp).double()

                for i in range(3):
                    x_axis[:,i] = torch.div(x_axis_temp[:,i],x_length)
                    y_axis[:,i] = torch.div(y_axis_temp[:,i],y_length)
                    z_axis[:,i] = torch.div(z_axis_temp[:,i],z_length)

                Rot_gt_temp = torch.zeros(len(gt_data),3,3).double()
                Rot_gt_temp[:,:,0] = x_axis
                Rot_gt_temp[:,:,1] = y_axis
                Rot_gt_temp[:,:,2] = z_axis

                ratio1 = 0.29
                ratio2 = 0.495
                p_gt_left = gt_back_left[:,0:2]+(gt_front_left[:,0:2] - gt_back_left[:,0:2])*ratio1
                p_gt_right = gt_back_right[:,0:2] + (gt_front_right[:,0:2] - gt_back_right[:,0:2])*ratio1

                p_gt_temp = torch.zeros(len(gt_data),3).double()
                p_gt_temp[:,0:2] = p_gt_left + (p_gt_right - p_gt_left) * ratio2
                p_gt_temp[:,2] = (gt_front_left[:,2]+gt_back_left[:,2]+gt_back_right[:,2]+gt_front_right[:,2])/4

                t_gt = torch.Tensor(gt_data[:,0] - gt_data[0,0]).double()

                p_gt_init = p_gt_temp[0].clone().detach()

                p_gt = p_gt_temp - p_gt_init

                Rot_gt_init = Rot_gt_temp[0].clone().detach()
                Rot_gt = mtbm(Rot_gt_init,Rot_gt_temp)
                dRot_gt = torch.zeros(len(gt_data),3,3).double()
                dRot_gt[0] = torch.eye(3).double()
                dRot_gt[1:] = bmtm(Rot_gt[:-1],Rot_gt[1:])
                dRot_gt = SO3.dnormalize(dRot_gt.cuda())
                # dRot_gt = dRot_gt.reshape(dRot_gt.size(0),9,)
                
                Rot_gt_recon = torch.zeros_like(dRot_gt).double()
                Rot_gt_recon[0] = dRot_gt[0]
                for i in range(1, Rot_gt_recon.size(0)):
                    Rot_gt_recon[i] = torch.mm(Rot_gt_recon[i-1],dRot_gt[i])

                ang_gt = SO3.to_rpy(Rot_gt)
                ang_gt_recon = SO3.to_rpy(Rot_gt_recon)
                quat_gt = SO3.to_quaternion(Rot_gt,ordering='xyzw')

                p_gt = mtbv(Rot_gt_init,p_gt)

                v_gt = torch.zeros_like(p_gt).double()
                v_gt[:-1] = (p_gt[1:]-p_gt[:-1])/0.01

                v_loc_gt = bmtv(Rot_gt, v_gt)

                Rot_gt = Rot_gt.reshape(Rot_gt.size(0),9,)

                # dt_gt = t_gt[1:] - t_gt[:-1]


            elif (date_dir == dataset_name+"_imu.csv"):
                path1 = os.path.join(path_data_base, date_dir)
                imu_csv = pd.read_csv(path1,sep=",")
                imu_data = imu_csv.to_numpy()

                R_imu_from_robot = torch.tensor([[0,1,0],
                                                    [1,0,0],
                                                    [0,0,-1]]).double()

                Rot_imu_temp = torch.zeros(len(imu_data),3,3).double()
                Rot_imu_temp[:,0,:] = torch.Tensor(imu_data[:,7:10].astype(np.float64)).double()
                Rot_imu_temp[:,1,:] = torch.Tensor(imu_data[:,10:13].astype(np.float64)).double()
                Rot_imu_temp[:,2,:] = torch.Tensor(imu_data[:,13:16].astype(np.float64)).double()

                acc_imu_temp = torch.Tensor(imu_data[:,22:25].astype(np.float64)).double()
                acc_imu = mbv(R_imu_from_robot,acc_imu_temp)

                t_imu = torch.Tensor(imu_data[:,2].astype(np.float64) - imu_data[0,2]).double()

                w_imu_temp = torch.Tensor(imu_data[:,26:29].astype(np.float64)).cuda().double()
                w_imu = mbv(R_imu_from_robot.cuda(), w_imu_temp)
                dRot_imu_recon = SO3.exp(0.01*w_imu)
                Rot_imu_recon = torch.zeros_like(dRot_imu_recon).double()
                # Rot_imu_recon[0] = torch.mm(torch.tensor([[1,0,0],
                #                                     [0,1,0],
                #                                     [0,0,-1]]).cuda().double(),dRot_imu_recon[0])
                Rot_imu_recon[0] = dRot_imu_recon[0]                                                    
                for i in range(1,Rot_imu_recon.size(0)):
                    Rot_imu_recon[i] = torch.mm(Rot_imu_recon[i-1], dRot_imu_recon[i])

                Rot_imu_init = Rot_imu_temp[0].clone().detach()
                Rot_imu = mtbm(Rot_imu_init,Rot_imu_temp)

                quat_imu = SO3.to_quaternion(Rot_imu, ordering='xyzw')

                ang_imu = SO3.to_rpy(Rot_imu)
                ang_imu_recon = SO3.to_rpy(Rot_imu_recon)
                # dt_imu = t_imu[1:] - t_imu[:-1]

            elif (date_dir == dataset_name+"-husky_velocity_controller-odom.csv"):
                path1 = os.path.join(path_data_base, date_dir)
                wheel_csv = pd.read_csv(path1,sep=",")
                wheel_data = wheel_csv.to_numpy()

                p_wheel = torch.Tensor(wheel_data[:,6:9].astype(np.float64)).double()
                
                #nonholonomic: v_wheel 1-dim
                v_wheel = torch.Tensor(wheel_data[:,14].astype(np.float64)).double().unsqueeze(1)
                ## holonomic: v_wheel 2-dim
                # v_wheel = torch.Tensor(wheel_data[:,14:16].astype(np.float64)).double()

                t_wheel = torch.Tensor(wheel_data[:,2].astype(np.float64) - wheel_data[0,2] + wheel_data[:,3].astype(np.float64)/1e9 - wheel_data[0,3]/1e9).double()
                
                quat_wheel = torch.Tensor(wheel_data[:,9:13].astype(np.float64)).double()

                Rot_wheel_temp = SO3.from_quaternion(quat_wheel,ordering='xyzw')
                Rot_wheel_init = Rot_wheel_temp[0].clone().detach()
                Rot_wheel = mtbm(Rot_wheel_init,Rot_wheel_temp)

                ang_wheel = SO3.to_rpy(Rot_wheel)

        v_wheel_joint_temp = interp_data(t_wheel.numpy(), v_wheel.numpy(), t_imu.numpy())

        mondict = { 't_gt': t_gt.numpy(), 't_imu': t_imu, 't_wheel': t_wheel, 'ang_gt': ang_gt.numpy(), 'ang_gt_recon': ang_gt_recon.cpu().detach().numpy(), 
                    'ang_imu': ang_imu.numpy(), 'ang_imu_recon': ang_imu_recon.cpu().detach().numpy(), 'v_loc_wheel': v_wheel_joint_temp, 'v_loc_gt': v_loc_gt.numpy(), 'v_wheel':v_wheel

        }
        
        dump(mondict, path_data_save, dataset_name + ".p")



        results_plot_ROT(path_data_save, path_results, dataset_name)    