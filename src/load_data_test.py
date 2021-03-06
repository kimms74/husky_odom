import numpy as np
import pandas as pd
import os
import math
import pickle
import copy
from result_plot_test import results_plot_test
from scipy.signal import filtfilt, butter
from scipy.spatial.transform import Rotation as R

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
    # datasets = ["square_ccw","square_cw","circle_ccw","circle_cw","ribbon","inf","random_1","random_2"]
    # datasets = ["origin4"]
    # datasets = ['move1', 'move2', 'move3', 'move4', 'move5', 'move6', 'move7', 'move8', 'move9', 'origin1', 'origin2', 'origin3','origin4']
    # datasets = ['origin4','move9']
    # datasets = ['move10', 'move11', 'move12', 'move13', 'move14', 'move15', 'move16', 'move17', 'move18', 'move19', 'move20', 'move21', 
    #             'origin5', 'origin6', 'origin7', 'origin8']
    datasets = ['origin8']
    for dataset_name in datasets:
        # path_data_base = "../test_dataset/220128/"+dataset_name +"/"
        # path_data_base = "../../../Datasets/husky_dataset/"+dataset_name +"/"
        path_data_base = "../../../Datasets/husky_dataset/"
        path_data_save = "../data"
        path_results = "../results"

        date_dirs = os.listdir(path_data_base)
        for n_iter, date_dir in enumerate(date_dirs):
            path1 = os.path.join(path_data_base, date_dir)
            date_dirs2 = os.listdir(path1)
            for n_iter2, date_dir2 in enumerate(date_dirs2):
                path2 = os.path.join(path1,date_dir2)
                dirs = os.listdir(path2)
                for _, dir in enumerate(dirs):
                    path3 = os.path.join(path2,dir)
                    if (dir == dataset_name+"_mocap.txt"):
                        gt_csv = pd.read_csv(path3,sep="\t")
                        gt_data = gt_csv.to_numpy()

                        p_gt_temp = np.zeros((len(gt_data),3))
                        p_gt = np.zeros((len(gt_data),3))
                        v_gt = np.zeros((len(gt_data),3))
                        v_loc_gt = np.zeros((len(gt_data),3))
                        t_gt = np.zeros(len(gt_data))

                        Rot_gt_temp = np.zeros((len(gt_data),3,3))
                        Rot_gt = np.zeros((len(gt_data),3,3))

                        x_axis = np.zeros((len(gt_data),3))
                        y_axis = np.zeros((len(gt_data),3))
                        z_axis = np.zeros((len(gt_data),3))

                        roll_gt = np.zeros(len(gt_data))
                        pitch_gt = np.zeros(len(gt_data))
                        yaw_gt = np.zeros(len(gt_data))

                        # if (date_dir == '211207'):
                        #     gt_front_left = gt_data[:,1:4]/1e3
                        #     gt_front_right = gt_data[:,4:7]/1e3
                        #     gt_back_left = gt_data[:,7:10]/1e3
                        #     gt_back_right = gt_data[:,10:13]/1e3

                        #     y_axis_temp = ((gt_front_left - gt_front_right) + (gt_back_left - gt_back_right))/2
                        #     x_axis_temp = ((gt_front_right - gt_back_right) + (gt_front_left - gt_back_left))/2

                        #     for i in range(len(gt_data)):
                        #         x_axis[i] = x_axis_temp[i]/(math.sqrt(math.pow(x_axis_temp[i,0],2)+math.pow(x_axis_temp[i,1],2)+math.pow(x_axis_temp[i,2],2)))
                        #         y_axis[i] = y_axis_temp[i]/(math.sqrt(math.pow(y_axis_temp[i,0],2)+math.pow(y_axis_temp[i,1],2)+math.pow(y_axis_temp[i,2],2)))
                        #         z_axis_temp = (skew(x_axis_temp[i]).dot(y_axis_temp[i]))
                        #         z_axis[i] = z_axis_temp/(math.sqrt(math.pow(z_axis_temp[0],2)+math.pow(z_axis_temp[1],2)+math.pow(z_axis_temp[2],2)))

                        #         Rot_gt_temp[i,:,0] = x_axis[i]
                        #         Rot_gt_temp[i,:,1] = y_axis[i]
                        #         Rot_gt_temp[i,:,2] = z_axis[i]
                
                        #         ratio1 = 0.29
                        #         ratio2 = 0.495
                        #         p_gt_left = gt_back_left[i,0:2]+(gt_front_left[i,0:2] - gt_back_left[i,0:2])*ratio1
                        #         p_gt_right = gt_back_right[i,0:2] + (gt_front_right[i,0:2] - gt_back_right[i,0:2])*ratio1
                        #         p_gt_temp[i,0:2] = p_gt_left + (p_gt_right - p_gt_left) * ratio2
                        #         p_gt_temp[i,2] = (gt_front_left[i,2]+gt_back_left[i,2]+gt_back_right[i,2]+gt_front_right[i,2])/4

                        # elif (date_dir == '220311' or date_dir == '220325'):
                        gt_front = gt_data[:,1:4]/1e3
                        gt_center = gt_data[:,4:7]/1e3
                        gt_left = gt_data[:,7:10]/1e3
                        gt_right = gt_data[:,10:13]/1e3

                        x_axis_temp = (gt_front - gt_center)
                        y_axis_temp = (gt_left - gt_center)

                        for i in range(len(gt_data)):
                            x_axis[i] = x_axis_temp[i]/(math.sqrt(math.pow(x_axis_temp[i,0],2)+math.pow(x_axis_temp[i,1],2)+math.pow(x_axis_temp[i,2],2)))
                            y_axis[i] = y_axis_temp[i]/(math.sqrt(math.pow(y_axis_temp[i,0],2)+math.pow(y_axis_temp[i,1],2)+math.pow(y_axis_temp[i,2],2)))
                            z_axis_temp = (skew(x_axis_temp[i]).dot(y_axis_temp[i]))
                            z_axis[i] = z_axis_temp/(math.sqrt(math.pow(z_axis_temp[0],2)+math.pow(z_axis_temp[1],2)+math.pow(z_axis_temp[2],2)))
                            y_axis[i] = (skew(z_axis[i]).dot(x_axis[i]))
                            y_axis[i] = y_axis[i]/(math.sqrt(math.pow(y_axis[i,0],2)+math.pow(y_axis[i,1],2)+math.pow(y_axis[i,2],2)))

                            Rot_gt_temp[i,:,0] = x_axis[i]
                            Rot_gt_temp[i,:,1] = y_axis[i]
                            Rot_gt_temp[i,:,2] = z_axis[i]

                        p_gt_temp = gt_center - x_axis*0.21
                        # p_gt_temp = gt_center

                        t_gt_temp = gt_data[:,0]
                        t_gt_init = copy.deepcopy(gt_data[0,0])
                        p_gt_init = copy.deepcopy(p_gt_temp[0])

                        p_gt = p_gt_temp - p_gt_init
                        t_gt = t_gt_temp - t_gt_init

                        Rot_gt_init = copy.deepcopy(Rot_gt_temp[0])
                        for i in range(len(gt_data)):
                            Rot_gt[i] = (Rot_gt_init.transpose()).dot(Rot_gt_temp[i])

                            roll_gt[i], pitch_gt[i], yaw_gt[i] = to_rpy(Rot_gt[i])

                            p_gt[i] = (Rot_gt_init.transpose()).dot(p_gt[i])


                            if i >=1:
                                v_gt[i] = (p_gt[i]-p_gt[i-1])/0.01

                        ang_gt = np.zeros((len(gt_data),3))
                        ang_gt[:, 0] = roll_gt
                        ang_gt[:, 1] = pitch_gt
                        ang_gt[:, 2] = yaw_gt

                        for i in range(len(gt_data)):
                            v_loc_gt[i] = (Rot_gt[i].transpose()).dot(v_gt[i])

                    elif (dir == dataset_name+"-imu-data.csv"):
                        imu_csv = pd.read_csv(path3,sep=",")
                        imu_data = imu_csv.to_numpy()

                        acc_imu_temp = np.zeros((len(imu_data),3))
                        acc_imu = np.zeros((len(imu_data),3))
                        acc_filter_imu =np.zeros((len(imu_data),3))
                        t_imu = np.zeros((len(imu_data)))
                        quat_imu = np.zeros((len(imu_data),4))
                        Rot_imu_temp = np.zeros((len(imu_data),3,3))
                        Rot_imu = np.zeros((len(imu_data),3,3))
                        ang_imu = np.zeros((len(imu_data),3))
                        gravity_vec = np.zeros((len(imu_data),3))
                        w_imu_temp = np.zeros((len(imu_data),3))
                        w_imu = np.zeros((len(imu_data),3))
                        dRot_imu = np.zeros((len(imu_data),3,3))

                        R_imu_from_robot = np.identity(3)
                        R_imu_from_robot[:,0] = (0,1,0)
                        R_imu_from_robot[:,1] = (1,0,0)
                        R_imu_from_robot[:,2] = (0,0,-1)

                        gravity = 9.80665
                        g_corr = 9.77275
                        g = np.array([0, 0, 9.80665])

                        # if (date_dir == '211207'):
                        #     w_imu_temp = imu_data[:,26:29]

                        #     for i in range(len(imu_data)):
                        #         w_imu[i] = R_imu_from_robot.dot(w_imu_temp[i])
                        #         acc_imu[i] = R_imu_from_robot.dot(imu_data[i,22:25])

                        #         t_imu[i] = imu_data[i,2] - imu_data[0,2]
                        #         dRot_imu[i] = so3exp(0.01*w_imu[i])

                        # elif (date_dir == '220311'):
                        w_imu_temp = imu_data[:,10:13]

                        for i in range(len(imu_data)):
                            w_imu[i] = R_imu_from_robot.dot(w_imu_temp[i])
                            acc_imu[i] = R_imu_from_robot.dot((imu_data[i,14:17]) + g)

                            t_imu[i] = (imu_data[i,2] + imu_data[i,3]/1e9) - (imu_data[0,2] + imu_data[0,3]/1e9) 
                        
                            dRot_imu[i] = so3exp(0.01*w_imu[i])

                        Rot_imu[0] = copy.deepcopy(dRot_imu[0])
                        acc_filter_imu = acc_imu

                        for i in range(1,len(imu_data)):
                            Rot_imu[i] = Rot_imu[i-1].dot(dRot_imu[i])

                        roll_imu = np.zeros(len(Rot_imu))
                        pitch_imu = np.zeros(len(Rot_imu))
                        yaw_imu = np.zeros(len(Rot_imu))
                        
                        for i in range(len(imu_data)):
                            roll_imu[i], pitch_imu[i], yaw_imu[i] = to_rpy(Rot_imu[i])
                            quat_imu[i] = (R.from_matrix(Rot_imu[i])).as_quat() #x,y,z,w
                    
                        ang_imu = np.zeros((len(Rot_imu),3))
                        ang_imu[:,0] = roll_imu
                        ang_imu[:,1] = pitch_imu
                        ang_imu[:,2] = yaw_imu

                        dt_imu = t_imu[1:] - t_imu[:-1]

                    elif (dir == dataset_name+"-husky_velocity_controller-odom.csv"):
                        wheel_csv = pd.read_csv(path3,sep=",")
                        wheel_data = wheel_csv.to_numpy()

                        p_wheel = np.zeros((len(wheel_data),3))
                        v_wheel = np.zeros((len(wheel_data),3))
                        t_wheel = np.zeros(len(wheel_data))
                        w_wheel = np.zeros((len(wheel_data),3))

                        quaternion = np.zeros((len(wheel_data),4))
                        Rot_wheel_temp = np.zeros((len(wheel_data),3,3))
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
                            # p_wheel[i] = R_husky_from_mocap.dot(p_wheel[i])
                            v_wheel[i] = wheel_data[i,14:17]
                            # v_wheel[i] = R_husky_from_mocap.dot(v_wheel[i])

                            t_wheel[i] = (wheel_data[i,2] + wheel_data[i,3]/1e9) - (wheel_data[0,2] + wheel_data[0,3]/1e9)
                            
                            quaternion[i] = wheel_data[i,9:13]
                            Rot_wheel_temp[i] = quaternion_to_rotation_matrix(quaternion[i])

                        Rot_wheel_init = copy.deepcopy(Rot_wheel_temp[0])

                        for i in range(len(wheel_data)):
                            Rot_wheel[i] = (Rot_wheel_init.transpose()).dot(Rot_wheel_temp[i])
                            roll_wheel[i], pitch_wheel[i], yaw_wheel[i] = to_rpy(Rot_wheel[i])

                        ang_wheel = np.zeros((len(wheel_data),3))
                        ang_wheel[:, 0] = roll_wheel
                        ang_wheel[:, 1] = pitch_wheel
                        ang_wheel[:, 2] = yaw_wheel

                        for i in range(len(wheel_data)):
                            t_wheel[i] = (wheel_data[i,2] + wheel_data[i,3]/1e9) - (wheel_data[0,2] + wheel_data[0,3]/1e9)

                    elif (dir == dataset_name+"-odometry-filtered.csv"):
                        ekf_csv = pd.read_csv(path3,sep=",")
                        ekf_data = ekf_csv.to_numpy()

                        p_ekf = np.zeros((len(ekf_data),3))
                        v_ekf = np.zeros((len(ekf_data),3))
                        v_loc_ekf = np.zeros((len(ekf_data),3))
                        t_ekf = np.zeros(len(ekf_data))
                        w_ekf = np.zeros((len(ekf_data),3))

                        quaternion = np.zeros((len(ekf_data),4))
                        Rot_ekf_temp = np.zeros((len(ekf_data),3,3))
                        Rot_ekf = np.zeros((len(ekf_data),3,3))

                        roll_ekf = np.zeros(len(ekf_data))
                        pitch_ekf = np.zeros(len(ekf_data))
                        yaw_ekf = np.zeros(len(ekf_data))
                        
                        for i in range(len(ekf_data)):
                            p_ekf[i] = ekf_data[i,6:9]
                            v_ekf[i] = ekf_data[i,14:17]

                            t_ekf[i] = (ekf_data[i,2] + ekf_data[i,3]/1e9) - (ekf_data[0,2] + ekf_data[0,3]/1e9)
                            
                            quaternion[i] = ekf_data[i,9:13]
                            Rot_ekf_temp[i] = quaternion_to_rotation_matrix(quaternion[i])

                        Rot_ekf_init = copy.deepcopy(Rot_ekf_temp[0])

                        for i in range(len(ekf_data)):
                            Rot_ekf[i] = (Rot_ekf_init.transpose()).dot(Rot_ekf_temp[i])
                            roll_ekf[i], pitch_ekf[i], yaw_ekf[i] = to_rpy(Rot_ekf[i])

                        ang_ekf = np.zeros((len(ekf_data),3))
                        ang_ekf[:, 0] = roll_ekf
                        ang_ekf[:, 1] = pitch_ekf
                        ang_ekf[:, 2] = yaw_ekf

                        for i in range(len(ekf_data)):
                            t_ekf[i] = (ekf_data[i,2] + ekf_data[i,3]/1e9) - (ekf_data[0,2] + ekf_data[0,3]/1e9)
                            v_loc_ekf[i] = (Rot_ekf[i].transpose()).dot(v_ekf[i])

        mondict_gt = {
            't_gt': t_gt, 'p_gt': p_gt, 'Rot_gt': Rot_gt,'ang_gt': ang_gt, 'v_gt': v_gt, 'v_loc_gt': v_loc_gt,
             'name': dataset_name, 't0': t_gt[0]
            }
        dump(mondict_gt, path_data_save, dataset_name + "_gt.p")

        v_imu, p_imu, b_omega_imu, b_acc_imu = run_imu(t_imu,Rot_imu, acc_filter_imu)
        
        mondict_imu = {
            't_imu': t_imu, 'Rot_imu': Rot_imu, 'ang_imu': ang_imu,'acc_imu': acc_filter_imu, 'v_imu': v_imu, 'p_imu': p_imu,
            'b_omega_imu': b_omega_imu, 'b_acc_imu': b_acc_imu
            }
        dump(mondict_imu, path_data_save, dataset_name + "_imu.p")

        v_wheel_joint = interp_data(t_wheel, v_wheel, t_imu)
        p_joint_imu, v_joint_imu = run_joint(t_imu,v_wheel_joint, Rot_imu)
        v_loc_joint_imu = np.zeros((len(imu_data),3))
        for i in range(len(imu_data)):
            v_loc_joint_imu[i] = (Rot_imu[i].transpose()).dot(v_joint_imu[i])

        mondict_joint = {
            't_joint_imu':t_imu, 'Rot_joint_imu':Rot_imu,'ang_joint_imu': ang_imu, 'p_joint_imu': p_joint_imu,
            'v_joint_imu': v_joint_imu, 'v_loc_joint_imu':v_loc_joint_imu
        }
        dump(mondict_joint, path_data_save, dataset_name + "_joint_imu.p")
        
        mondict_wheel = {
            't_wheel':t_wheel, 'Rot_wheel':Rot_wheel,'ang_wheel': ang_wheel, 'p_wheel': p_wheel,
            'v_wheel': v_wheel
        }
        dump(mondict_wheel, path_data_save, dataset_name + "_wheel.p")

        # mondict_ekf = {
        #     't_ekf':t_ekf, 'Rot_ekf':Rot_ekf,'ang_ekf': ang_ekf, 'p_ekf': p_ekf,
        #     'v_ekf': v_ekf, 'v_loc_ekf': v_loc_ekf
        # }
        # dump(mondict_ekf, path_data_save, dataset_name + "_ekf.p")



        results_plot_test(path_data_save, path_results, dataset_name)