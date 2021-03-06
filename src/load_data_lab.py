import numpy as np
import pandas as pd
import os
import math
import pickle
import copy

from result_plot_lab import *


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

    roll_imu = np.zeros(len(Rot_imu))
    pitch_imu = np.zeros(len(Rot_imu))
    yaw_imu = np.zeros(len(Rot_imu))

    for i in range(N):
        roll_imu[i], pitch_imu[i], yaw_imu[i] = to_rpy(Rot_imu[i])

    ang_imu = np.zeros((len(Rot_imu),3))
    ang_imu[:,0] = roll_imu
    ang_imu[:,1] = pitch_imu
    ang_imu[:,2] = yaw_imu

    for i in range(1,N):
        v_imu[i], p_imu[i], b_omega_imu[i], b_acc_imu[i] = propagate_imu(Rot_imu[i-1], v_imu[i-1], p_imu[i-1], b_omega_imu[i-1], b_acc_imu[i-1], acc_imu[i], dt_imu[i-1])

    return Rot_imu, ang_imu, v_imu, p_imu, b_omega_imu, b_acc_imu


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
    acc_joint = np.zeros((N, 3))

    for i in range(1,N):
        acc_joint[i] = (v_wheel_joint[i]-v_wheel_joint[i-1])/dt_joint[i-1]
        p_joint[i], v_joint[i] = propagate_joint(Rot_joint[i-1], p_joint[i-1], v_wheel_joint[i-1],acc_joint[i], dt_joint[i-1])

    return p_joint, v_joint


def propagate_joint(Rot_prev, p_prev, v_prev, a, dt):
    acc = Rot_prev.dot(a)
    v = Rot_prev.dot(v_prev) + acc*dt
    p = p_prev + v*dt + 0.5*acc*(dt**2)
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
    for i in range(100):
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
        # for j in range(3):
        #     if (abs(acc_imu[i,j]) < cut_value):
        #         a[i,j] = 0
        #     else : a[i,j] = acc_imu[i,j]
        
        if ((abs(acc_imu[i,1]) < cut_value[0]) or (abs(acc_imu[i,1]) > cut_value[1])):
            a[i,1] = 0
        else : a[i,1] = acc_imu[i,1]
        
    
    return a

if __name__ == '__main__':
    datasets = ["lab"]
    for dataset_name in datasets:
        # dataset_name = "fast_zigzag_left"
        # dataset_name = "fast_square_ccw"
        path_data_base = "../../Datasets/husky_dataset/"+dataset_name +"/"
        path_data_save = "data"
        path_results = "results"

        date_dirs = os.listdir(path_data_base)
        for n_iter, date_dir in enumerate(date_dirs):
            if (date_dir == dataset_name+".txt"):
                path1 = os.path.join(path_data_base, date_dir)
                gt_csv = pd.read_csv(path1,sep="\t")
                gt_data = gt_csv.to_numpy()

                gt_front_left = gt_data[:,1:4]/1e3
                gt_back_left = gt_data[:,4:7]/1e3
                gt_back_right = gt_data[:,7:10]/1e3
                gt_front_right = gt_data[:,10:13]/1e3

                x_init = (gt_front_left[0,0]+gt_back_left[0,0]+gt_back_right[0,0]+gt_front_right[0,0])/4
                y_init = (gt_front_left[0,1]+gt_back_left[0,1]+gt_back_right[0,1]+gt_front_right[0,1])/4
                z_init = (gt_front_left[0,2]+gt_back_left[0,2]+gt_back_right[0,2]+gt_front_right[0,2])/4

                p_gt = np.zeros((len(gt_back_left),3))
                v_gt = np.zeros((len(gt_back_left),3))
                t_gt = np.zeros(len(gt_back_left))

                Rot_gt = np.zeros((len(gt_back_left),3,3))

                x_axis_temp = ((gt_front_right - gt_front_left) + (gt_back_right - gt_back_left))/2
                y_axis_temp = ((gt_front_right - gt_back_right) + (gt_front_left - gt_back_left))/2

                x_axis = np.zeros((len(gt_back_left),3))
                y_axis = np.zeros((len(gt_back_left),3))
                z_axis = np.zeros((len(gt_back_left),3))

                roll_gt = np.zeros(len(gt_front_left))
                pitch_gt = np.zeros(len(gt_front_left))
                yaw_gt = np.zeros(len(gt_front_left))

                for i in range(len(gt_back_left)):
                    x_axis[i] = x_axis_temp[i]/(math.sqrt(math.pow(x_axis_temp[i,0],2)+math.pow(x_axis_temp[i,1],2)+math.pow(x_axis_temp[i,2],2)))
                    y_axis[i] = y_axis_temp[i]/(math.sqrt(math.pow(y_axis_temp[i,0],2)+math.pow(y_axis_temp[i,1],2)+math.pow(y_axis_temp[i,2],2)))
                    z_axis_temp = (skew(x_axis_temp[i]).dot(y_axis_temp[i]))
                    z_axis[i] = z_axis_temp/(math.sqrt(math.pow(z_axis_temp[0],2)+math.pow(z_axis_temp[1],2)+math.pow(z_axis_temp[2],2)))

                    Rot_gt[i,:,0] = x_axis[i]
                    Rot_gt[i,:,1] = y_axis[i]
                    Rot_gt[i,:,2] = z_axis[i]

                    p_gt[i,0] = (gt_front_left[i,0]+gt_back_left[i,0]+gt_back_right[i,0]+gt_front_right[i,0])/4 - x_init
                    p_gt[i,1] = (gt_front_left[i,1]+gt_back_left[i,1]+gt_back_right[i,1]+gt_front_right[i,1])/4 - y_init
                    p_gt[i,2] = (gt_front_left[i,2]+gt_back_left[i,2]+gt_back_right[i,2]+gt_front_right[i,2])/4 - z_init
                    t_gt[i] = gt_data[i,0] - gt_data[0,0]

                    roll_gt[i], pitch_gt[i], yaw_gt[i] = to_rpy(Rot_gt[i])
                    if i >=1:
                        v_gt[i] = (p_gt[i]-p_gt[i-1])/0.01

                ang_gt = np.zeros((len(gt_back_left),3))
                ang_gt[:, 0] = roll_gt - roll_gt[0]
                ang_gt[:, 1] = pitch_gt - pitch_gt[0]
                ang_gt[:, 2] = yaw_gt - yaw_gt[0]

            elif (date_dir == dataset_name+"-gx4-filter-imu-data.csv"):
                path1 = os.path.join(path_data_base, date_dir)
                imu_csv = pd.read_csv(path1,sep=",")
                imu_data = imu_csv.to_numpy()

                acc_imu = np.zeros((len(imu_data),3))
                acc_filter_imu =np.zeros((len(imu_data),3))
                t_imu = np.zeros((len(imu_data)))
                quaternion = np.zeros((len(imu_data),4)) # w,x,y,z
                Rot_imu = np.zeros((len(imu_data),3,3))
                ang_imu = np.zeros((len(imu_data),3))
                gravity_vec = np.zeros((len(imu_data),3))
                gravity = np.zeros(len(imu_data))

                R_imu_from_mocap = np.identity(3)
                R_imu_from_mocap[:,0] = (-1,0,0)
                R_imu_from_mocap[:,1] = (0,1,0)
                R_imu_from_mocap[:,2] = (0,0,-1)

                # g_corr = 9.77275
                g_corr = 9.8
                # g_corr = 11

                for i in range(len(imu_data)):
                    quaternion[i] = imu_data[i,7:11] #w,x,y,z
                    for j in range(3):
                        # Rot_imu[i,j,:] = R_imu_from_mocap.dot(imu_data[i,(20+3*j):(23+3*j)])
                        Rot_imu[i,j,:] = imu_data[i,(20+3*j):(23+3*j)]
                    ang_imu[i] = imu_data[i,3:6]
                    gravity[i] = imu_data[i,30]
                    gravity_vec[i] = imu_data[i,16:19]
                    # acc_imu[i] = R_imu_from_mocap.dot(imu_data[i,12:15] - (gravity[i] - g_corr)/(gravity[i])*gravity_vec[i])
                    acc_imu[i] = R_imu_from_mocap.dot(imu_data[i,12:15])
                    t_imu[i] = imu_data[i,2] - imu_data[0,2]
                
                cut_value = [0.16,1.67]
                acc_filter_imu = high_pass_filter(acc_imu, cut_value)
                # acc_filter_imu = acc_imu
                Rot_imu_init = copy.deepcopy(Rot_imu[0])

                for i in range(len(imu_data)):
                    Rot_imu[i] = (Rot_imu_init.transpose()).dot(Rot_imu[i])

            elif (date_dir == dataset_name+"-joint_states.csv"):
                path1 = os.path.join(path_data_base, date_dir)
                joint_csv = pd.read_csv(path1,sep=",")
                joint_data = joint_csv.to_numpy()
                
                v_wheel_joint = np.zeros((len(joint_data),3))
                t_joint = np.zeros((len(joint_data)))

                for i in range(len(joint_data)):
                    v_wheel_joint[i,0] = (float(((joint_data[i,7])[1:-1].split(', '))[0])+float(((joint_data[i,7])[1:-1].split(', '))[1]))*0.1651/4
                    t_joint[i] = (joint_data[i,2] + joint_data[i,3]/1e9) - (joint_data[0,2] + joint_data[0,3]/1e9)
                    v_wheel_joint[i] = R_husky_from_mocap.dot(v_wheel_joint[i])
            
            elif (date_dir == dataset_name+"-husky_velocity_controller-odom.csv"):
                
                path1 = os.path.join(path_data_base, date_dir)
                wheel_csv = pd.read_csv(path1,sep=",")
                wheel_data = wheel_csv.to_numpy()

                p_wheel = np.zeros((len(wheel_data),3))
                v_wheel = np.zeros((len(wheel_data),3))
                t_wheel = np.zeros(len(wheel_data))
                w_wheel = np.zeros((len(wheel_data),3))

                quaternion = np.zeros((len(wheel_data),4))
                Rot_wheel = np.zeros((len(wheel_data),3,3))

                roll_wheel = np.zeros(len(wheel_data))
                pitch_wheel = np.zeros(len(wheel_data))
                yaw_wheel = np.zeros(len(wheel_data))

                R_husky_from_mocap = np.identity(3)
                R_husky_from_mocap[:,0] = (0,1,0)
                R_husky_from_mocap[:,1] = (-1,0,0)
                R_husky_from_mocap[:,2] = (0,0,1)
                
                for i in range(len(wheel_data)):
                    p_wheel[i] = wheel_data[i,6:9]
                    p_wheel[i] = R_husky_from_mocap.dot(p_wheel[i])
                    v_wheel[i] = wheel_data[i,14:17]
                    v_wheel[i] = R_husky_from_mocap.dot(v_wheel[i])

                    t_wheel[i] = (wheel_data[i,2] + wheel_data[i,3]/1e9) - (wheel_data[0,2] + wheel_data[0,3]/1e9)
                    
                    quaternion[i] = wheel_data[i,9:13]
                    Rot_wheel[i] = quaternion_to_rotation_matrix(quaternion[i])

                    roll_wheel[i], pitch_wheel[i], yaw_wheel[i] = to_rpy(Rot_wheel[i])

                ang_wheel = np.zeros((len(wheel_data),3))
                ang_wheel[:, 0] = roll_wheel
                ang_wheel[:, 1] = pitch_wheel
                ang_wheel[:, 2] = yaw_wheel

                for i in range(len(wheel_data)):
                    t_wheel[i] = (wheel_data[i,2] + wheel_data[i,3]/1e9) - (wheel_data[0,2] + wheel_data[0,3]/1e9)

        # mondict_gt = {
        #     't_gt': t_gt, 'p_gt': p_gt, 'Rot_gt': Rot_gt,'ang_gt': ang_gt, 'v_gt': v_gt,
        #     'name': dataset_name, 't0': t_gt[0]
        #     }
        # dump(mondict_gt, path_data_save, dataset_name + "_gt.p")

        Rot_imu, ang_imu, v_imu, p_imu, b_omega_imu, b_acc_imu = run_imu(t_imu,Rot_imu, acc_filter_imu)
        mondict_imu = {
            't_imu': t_imu, 'Rot_imu': Rot_imu, 'ang_imu': ang_imu, 'acc_imu': acc_filter_imu, 'v_imu': v_imu, 'p_imu': p_imu,
            'b_omega_imu': b_omega_imu, 'b_acc_imu': b_acc_imu
            }
        dump(mondict_imu, path_results, dataset_name + "_imu.p")

        p_joint, v_joint = run_joint(t_joint,v_wheel_joint, Rot_wheel)

        mondict_joint = {
            't_joint':t_joint, 'Rot_joint':Rot_wheel,'ang_joint': ang_wheel, 'p_joint': p_joint,
            'v_joint': v_joint
        }
        dump(mondict_joint, path_results, dataset_name + "_joint.p")

        mondict_wheel = {
            't_wheel':t_wheel, 'Rot_wheel':Rot_wheel,'ang_wheel': ang_wheel, 'p_wheel': p_wheel,
            'v_wheel': v_wheel
        }
        dump(mondict_wheel, path_results, dataset_name + "_wheel.p")



        results_plot_lab(path_data_save, path_results, dataset_name)    