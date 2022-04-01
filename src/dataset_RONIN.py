from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import pandas as pd
import math
import copy
from scipy.spatial.transform import Rotation as R
from utils import *
from scipy.ndimage import gaussian_filter1d

class HuskyRONINData:

    feature_dim = 8 # wheel_speed, accel, quat_imu (1,3,4)
    # target_dim = 7 # val_gt, quat_gt
    target_dim = 2 # val_gt
    # aux_dim = 4 # dt_gt, p_gt
    aux_dim = 7 # dt_gt, p_gt, quat_gt

    def __init__(self, path_data_base, path_data_save, data_list, mode, read_from_data):
        # paths
        self.path_data_base = path_data_base
        """path where raw data are saved"""
        self.path_data_save = path_data_save
        """path where data are saved"""
        self.data_list = data_list
        """data list"""   

        self.mode = mode
        self.read_from_data = read_from_data

        if self.read_from_data == True:
            self.read_data()
        else:
            self.load_data()

    def load(self, *_file_name):
        pickle_extension = ".p"
        file_name = os.path.join(*_file_name)
        if not file_name.endswith(pickle_extension):
            file_name += pickle_extension
        with open(file_name, "rb") as file_pi:
            pickle_dict = pickle.load(file_pi)
        return pickle_dict

    def load_data(self):
        for n_iter, dataset_name in enumerate(self.data_list):
            file_name = os.path.join(self.path_data_save+"/", dataset_name + "_" + self.mode + ".p")
            if not os.path.exists(file_name):
                print('No result for ' + dataset_name)
                return

            mondict = self.load(file_name)
            v_wheel_joint = mondict['v_wheel_joint']
            acc_imu = mondict['acc_imu']
            quat_imu = mondict['quat_imu']
            v_gt = mondict['v_gt']
            quat_gt = mondict['quat_gt']
            dt_gt = mondict['dt_gt']
            p_gt = mondict['p_gt']

            # features = np.concatenate([v_wheel_joint, acc_imu, quat_imu],axis=1)
            features = np.concatenate([v_wheel_joint, acc_imu, quat_imu],axis=1)
            # self.targets = np.concatenate([v_gt[:, :2],quat_gt],axis=1)
            # targets = np.concatenate([v_gt,quat_gt],axis=1)
            targets = v_gt
            # aux = np.concatenate([dt_gt, p_gt], axis=1)
            aux = np.concatenate([dt_gt, p_gt, quat_gt], axis=1)

            self.features_all.append(features)
            self.targets_all.append(targets)
            self.aux_all.append(aux)

    def read_data(self):

        print('Start read_data')
        
        self.features_all, self.targets_all, self.aux_all = [], [], []

        for n_iter, dataset in enumerate(self.data_list):
            # get access to each sequence
            path1 = os.path.join(self.path_data_base, dataset)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:

                if (date_dir2 == dataset+"_mocap.txt"):
                    
                    path2 = os.path.join(path1, date_dir2)
                    gt_csv = pd.read_csv(path2,sep="\t")
                    gt_data = gt_csv.to_numpy()

                    gt_front_left = gt_data[:,1:4]/1e3
                    gt_front_right = gt_data[:,4:7]/1e3
                    gt_back_left = gt_data[:,7:10]/1e3
                    gt_back_right = gt_data[:,10:13]/1e3

                    p_gt_temp = np.zeros((len(gt_back_left),3))
                    p_gt = np.zeros((len(gt_back_left),3))
                    v_gt = np.zeros((len(gt_back_left),3))
                    speed_gt = np.zeros(len(gt_back_left))
                    t_gt = np.zeros(len(gt_back_left))

                    Rot_gt_temp = np.zeros((len(gt_back_left),3,3))
                    Rot_gt = np.zeros((len(gt_back_left),3,3))
                    quat_gt = np.zeros((len(gt_front_left),4))

                    # x_axis_temp = ((gt_front_right - gt_front_left) + (gt_back_right - gt_back_left))/2
                    # y_axis_temp = ((gt_front_right - gt_back_right) + (gt_front_left - gt_back_left))/2
                    y_axis_temp = ((gt_front_left - gt_front_right) + (gt_back_left - gt_back_right))/2
                    x_axis_temp = ((gt_front_right - gt_back_right) + (gt_front_left - gt_back_left))/2

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

                        Rot_gt_temp[i,:,0] = x_axis[i]
                        Rot_gt_temp[i,:,1] = y_axis[i]
                        Rot_gt_temp[i,:,2] = z_axis[i]
        
                        ratio1 = 0.29
                        ratio2 = 0.495
                        p_gt_left = gt_back_left[i,0:2]+(gt_front_left[i,0:2] - gt_back_left[i,0:2])*ratio1
                        p_gt_right = gt_back_right[i,0:2] + (gt_front_right[i,0:2] - gt_back_right[i,0:2])*ratio1
                        p_gt_temp[i,0:2] = p_gt_left + (p_gt_right - p_gt_left) * ratio2
                        p_gt_temp[i,2] = (gt_front_left[i,2]+gt_back_left[i,2]+gt_back_right[i,2]+gt_front_right[i,2])/4
                        t_gt[i] = gt_data[i,0] - gt_data[0,0]

                    p_gt_init = copy.deepcopy(p_gt_temp[0])
                    for i in range(len(gt_back_left)):
                        p_gt[i] = p_gt_temp[i] - p_gt_init

                    Rot_gt_init = copy.deepcopy(Rot_gt_temp[0])
                    for i in range(len(gt_back_left)):
                        Rot_gt[i] = (Rot_gt_init.transpose()).dot(Rot_gt_temp[i])

                        roll_gt[i], pitch_gt[i], yaw_gt[i] = to_rpy(Rot_gt[i])
                        quat_gt[i] = (R.from_matrix(Rot_gt[i])).as_quat() #x,y,z,w

                        p_gt[i] = (Rot_gt_init.transpose()).dot(p_gt[i])


                        if i >=1:
                            v_gt[i] = (p_gt[i]-p_gt[i-1])/0.01
                            speed_gt[i] = math.sqrt(math.pow(v_gt[i,0],2)+math.pow(v_gt[i,1],2)+math.pow(v_gt[i,2],2))

                    ang_gt = np.zeros((len(gt_back_left),3))
                    ang_gt[:, 0] = roll_gt
                    ang_gt[:, 1] = pitch_gt
                    ang_gt[:, 2] = yaw_gt

                    dt_gt = t_gt[1:] - t_gt[:-1]


                elif (date_dir2 == dataset+"_imu.csv"):
                    
                    path2 = os.path.join(path1, date_dir2)
                    imu_csv = pd.read_csv(path2,sep=",")
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

                    R_imu_from_mocap = np.identity(3)
                    R_imu_from_mocap[:,0] = (-1,0,0)
                    R_imu_from_mocap[:,1] = (0,1,0)
                    R_imu_from_mocap[:,2] = (0,0,-1)

                    R_imu_from_robot = np.identity(3)
                    R_imu_from_robot[:,0] = (0,1,0)
                    R_imu_from_robot[:,1] = (1,0,0)
                    R_imu_from_robot[:,2] = (0,0,-1)

                    gravity = 9.80665
                    g_corr = 9.77275

                    for i in range(len(imu_data)):
                        for j in range(3):
                            Rot_imu_temp[i,j,:] = imu_data[i,(7+3*j):(10+3*j)]
                        gravity_vec[i] = imu_data[i,30:33]
                        # acc_imu[i] = R_imu_from_robot.dot(imu_data[i,22:25] - ((gravity - g_corr)/gravity)*gravity_vec[i])
                        # acc_imu_temp[i,0:2] = imu_data[i,22:24] - (((gravity - g_corr)/gravity)*gravity_vec[i])[0:2]
                        acc_imu_temp[i,0:3] = imu_data[i,22:25]
                        acc_imu[i] = R_imu_from_robot.dot(acc_imu_temp[i])

                        t_imu[i] = imu_data[i,2] - imu_data[0,2]
                    
                    # cut_value = [0.26,5]
                    # acc_filter_imu = high_pass_filter(acc_imu, cut_value)
                    acc_filter_imu = acc_imu
                    Rot_imu_init = copy.deepcopy(Rot_imu_temp[0])

                    for i in range(len(imu_data)):
                        Rot_imu[i] = (Rot_imu_init.transpose()).dot(Rot_imu_temp[i])

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


                elif (date_dir2 == dataset+"-husky_velocity_controller-odom.csv"):

                    path2 = os.path.join(path1, date_dir2)
                    wheel_csv = pd.read_csv(path2,sep=",")
                    wheel_data = wheel_csv.to_numpy()

                    p_wheel = np.zeros((len(wheel_data),3))
                    v_wheel = np.zeros((len(wheel_data),1))
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
                        v_wheel[i] = wheel_data[i,14]
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
                    
            v_wheel_joint = interp_data(t_wheel, v_wheel, t_imu)

            features = np.concatenate([v_wheel_joint, acc_imu, quat_imu],axis=1)
            # self.targets = np.concatenate([v_gt[:, :2],quat_gt],axis=1)
            # targets = np.concatenate([v_gt,quat_gt],axis=1)
            targets = v_gt[:,:2]
            # aux = np.concatenate([dt_gt, p_gt], axis=1)
            aux = np.concatenate([t_gt[:, None], p_gt, quat_gt], axis=1)

            mondict = {
            't_gt': t_gt, 'p_gt': p_gt, 'Rot_gt': Rot_gt,'ang_gt': ang_gt, 'v_gt': v_gt,
                'quat_gt': quat_gt, 't_imu': t_imu, 'quat_imu': quat_imu, 'acc_imu': acc_imu, 'v_wheel_joint': v_wheel_joint,'features': features, 'targets': targets, 'aux': aux 
                }
            dump(mondict, self.path_data_save, dataset + "_" + self.mode + ".p")
            
            self.features_all.append(features)
            self.targets_all.append(targets)
            self.aux_all.append(aux)
                


class HUSKYResNet(Dataset):
    def __init__(self, data, random_shift, shuffle, args):
        super().__init__()

        self.feature_dim = data.feature_dim
        self.target_dim = data.target_dim
        self.aux_dim = data.aux_dim
        self.step_size = args.step_size
        self.window_size = args.window_size
        self.random_shift = random_shift
        self.shuffle = shuffle

        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []

        self.features = data.features_all
        self.targets = data.targets_all
        self.aux = data.aux_all

        for i in range(len(data.data_list)):
            self.ts.append(self.aux[i][:,0])
            self.orientations.append(self.aux[i][:,4:8])
            self.gt_pos.append(self.aux[i][:,1:4])
            self.index_map += [[i, j] for j in range(self.window_size + self.random_shift, self.targets[i].shape[0], self.step_size)]

        if self.shuffle == True:
            random.shuffle(self.index_map)


    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))
        # else:
        #     frame_id = max(self.window_size, frame_id)

        feat = self.features[seq_id][frame_id - self.window_size : frame_id]
        # feat = self.features[seq_id][frame_id : frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

class HUSKYLSTM(Dataset):
    def __init__(self, data, random_shift, shuffle, args):
        super().__init__()

        self.feature_dim = data.feature_dim
        self.target_dim = data.target_dim
        self.aux_dim = data.aux_dim
        self.step_size = args.step_size
        self.window_size = args.window_size
        self.random_shift = random_shift
        self.shuffle = shuffle

        self.index_map = []

        # # Optionally smooth the sequence
        # feat_sigma = args.feature_sigma
        # targ_sigma = args.target_sigma
        # if feat_sigma > 0:
        #     self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        # if targ_sigma > 0:
        #     self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        # max_norm = args.max_velocity_norm
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []

        self.features = data.features_all
        self.targets = data.targets_all
        self.aux = data.aux_all
        
        for i in range(len(data.data_list)):
            # self.features[i] = self.features[i][:-1]
            # self.targets[i] = self.targets[i]
            # self.ts.append(self.aux[i][:-1, :1])
            # self.orientations.append(self.aux[i][:-1, 4:8])
            # self.gt_pos.append(self.aux[i][:-1, 1:4])

            self.features[i] = self.features[i]
            self.targets[i] = self.targets[i]
            self.ts.append(self.aux[i][:, 0])
            self.orientations.append(self.aux[i][:, 4:8])
            self.gt_pos.append(self.aux[i][:, 1:4])

            # velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            # bad_data = velocity > max_norm
            for j in range(self.window_size + self.random_shift, self.targets[i].shape[0], self.step_size):
                # if not bad_data[j - self.window_size - self.random_shift:j + self.random_shift].any():
                    # self.index_map.append([i, j])
                self.index_map.append([i, j])

        if self.shuffle == True:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])
        
        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)