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
from lie_algebra import SO3

class HuskyData:

    feature_dim = 7 # nonholonomic_velocity, w_imu, acc_imu (1,3,3)
    target_dim = 5 # 2d_v_loc_gt, ang_vel_gt (2,3)

    def __init__(self, path_data_base, path_data_save, data_list, mode, read_from_data, dt):
        # paths
        self.path_data_base = path_data_base
        """path where raw data are saved"""
        self.path_data_save = path_data_save
        """path where data are saved"""
        self.data_list = data_list
        """data list"""   

        self.features_all, self.targets_all, self.aux_all = [], [], []

        self.mode = mode
        self.read_from_data = read_from_data
        self.dt = dt

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
            features = mondict['features']
            targets = mondict['targets']
            aux = mondict['aux']

            self.features_all.append(features)
            self.targets_all.append(targets)
            self.aux_all.append(aux)

    def read_data(self):

        print('Start read_data')
        
        for n_iter, dataset in enumerate(self.data_list):
            # get access to each sequence
            if not os.path.isdir(self.path_data_base):
                continue

            date_dirs = os.listdir(self.path_data_base)
            for date_dir in date_dirs:
                path1 = os.path.join(self.path_data_base, date_dir)
                date_dirs2 = os.listdir(path1)
                for date_dir2 in date_dirs2:
                    path2 = os.path.join(path1,date_dir2)
                    dirs = os.listdir(path2)
                    for dir in dirs:
                        path3 = os.path.join(path2,dir)
                        if (dir == dataset+"_mocap.txt"):
                            gt_csv = pd.read_csv(path3,sep="\t")
                            gt_data = gt_csv.to_numpy()

                            # if (date_dir == '211207'):
                            #     gt_front_left = torch.Tensor(gt_data[:,1:4]/1e3).double()
                            #     gt_front_right = torch.Tensor(gt_data[:,4:7]/1e3).double()
                            #     gt_back_left = torch.Tensor(gt_data[:,7:10]/1e3).double()
                            #     gt_back_right = torch.Tensor(gt_data[:,10:13]/1e3).double()

                            #     y_axis_temp = ((gt_front_left - gt_front_right) + (gt_back_left - gt_back_right))/2
                            #     x_axis_temp = ((gt_front_right - gt_back_right) + (gt_front_left - gt_back_left))/2
                            #     z_axis_temp = bmv(SO3.wedge(x_axis_temp),y_axis_temp)

                            #     ratio1 = 0.29
                            #     ratio2 = 0.495
                            #     p_gt_left = gt_back_left[:,0:2]+(gt_front_left[:,0:2] - gt_back_left[:,0:2])*ratio1
                            #     p_gt_right = gt_back_right[:,0:2] + (gt_front_right[:,0:2] - gt_back_right[:,0:2])*ratio1

                            #     p_gt_temp = torch.zeros(len(gt_data),3).double()
                            #     p_gt_temp[:,0:2] = p_gt_left + (p_gt_right - p_gt_left) * ratio2
                            #     p_gt_temp[:,2] = (gt_front_left[:,2]+gt_back_left[:,2]+gt_back_right[:,2]+gt_front_right[:,2])/4

                            #     x_length = torch.sqrt(torch.pow(x_axis_temp[:,0],2) + torch.pow(x_axis_temp[:,1],2) + torch.pow(x_axis_temp[:,2],2))
                            #     y_length = torch.sqrt(torch.pow(y_axis_temp[:,0],2) + torch.pow(y_axis_temp[:,1],2) + torch.pow(y_axis_temp[:,2],2))
                            #     z_length = torch.sqrt(torch.pow(z_axis_temp[:,0],2) + torch.pow(z_axis_temp[:,1],2) + torch.pow(z_axis_temp[:,2],2))

                            #     x_axis = torch.zeros_like(x_axis_temp).double()
                            #     y_axis = torch.zeros_like(y_axis_temp).double()
                            #     z_axis = torch.zeros_like(z_axis_temp).double()

                            #     for i in range(3):
                            #         x_axis[:,i] = torch.div(x_axis_temp[:,i],x_length)
                            #         y_axis[:,i] = torch.div(y_axis_temp[:,i],y_length)
                            #         z_axis[:,i] = torch.div(z_axis_temp[:,i],z_length)

                            #     Rot_gt_temp = torch.zeros(len(gt_data),3,3).double()
                            #     Rot_gt_temp[:,:,0] = x_axis
                            #     Rot_gt_temp[:,:,1] = y_axis
                            #     Rot_gt_temp[:,:,2] = z_axis

                            # elif (date_dir == '220311'):
                            gt_front = torch.Tensor(gt_data[:,1:4]/1e3).double()
                            gt_center = torch.Tensor(gt_data[:,4:7]/1e3).double()
                            gt_left = torch.Tensor(gt_data[:,7:10]/1e3).double()
                            gt_right = torch.Tensor(gt_data[:,10:13]/1e3).double()

                            x_axis_temp = gt_front - gt_center
                            y_axis_temp = gt_left - gt_center
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

                            ratio = 0.21
                            p_gt_temp = gt_center - x_axis*ratio

                            t_gt = torch.Tensor(gt_data[:,0] - gt_data[0,0]).double()

                            p_gt_init = p_gt_temp[0].clone().detach()

                            p_gt = p_gt_temp - p_gt_init

                            Rot_gt_init = Rot_gt_temp[0].clone().detach()
                            Rot_gt = mtbm(Rot_gt_init,Rot_gt_temp)
                            dRot_gt = torch.zeros(len(gt_data),3,3).double()
                            dRot_gt[0] = torch.eye(3).double()
                            dRot_gt[1:] = bmtm(Rot_gt[:-1],Rot_gt[1:])
                            dRot_gt = SO3.dnormalize(dRot_gt.cuda())
                            w_gt = (SO3.log(dRot_gt)/self.dt).double().cpu().detach()

                            dRot_gt = dRot_gt.reshape(dRot_gt.size(0),9,).cpu()
                            
                            rpy_gt = SO3.to_rpy(Rot_gt)

                            quat_gt = SO3.to_quaternion(Rot_gt,ordering='xyzw')

                            p_gt = mtbv(Rot_gt_init,p_gt)

                            v_gt = torch.zeros_like(p_gt).double()
                            v_gt[:-1] = (p_gt[1:]-p_gt[:-1])/0.01

                            v_loc_gt = bmtv(Rot_gt, v_gt)

                            Rot_gt = Rot_gt.reshape(Rot_gt.size(0),9,)

                            # dt_gt = t_gt[1:] - t_gt[:-1]




                        elif (dir == dataset+"-imu-data.csv"):
                            imu_csv = pd.read_csv(path3,sep=",")
                            imu_data = imu_csv.to_numpy()

                            R_imu_from_robot = torch.tensor([[0,1,0],
                                                            [1,0,0],
                                                            [0,0,-1]]).double()

                            # if (date_dir == '211207'):
                            #     # Rot_imu_temp = torch.zeros(len(imu_data),3,3).double()
                            #     # Rot_imu_temp[:,0,:] = torch.Tensor(imu_data[:,7:10].astype(np.float64)).double()
                            #     # Rot_imu_temp[:,1,:] = torch.Tensor(imu_data[:,10:13].astype(np.float64)).double()
                            #     # Rot_imu_temp[:,2,:] = torch.Tensor(imu_data[:,13:16].astype(np.float64)).double()

                            #     acc_imu_temp = torch.Tensor(imu_data[:,22:25].astype(np.float64)).double()
                            #     w_imu_temp = torch.Tensor(imu_data[:,26:29].astype(np.float64)).double()
                            #     t_imu = torch.Tensor(imu_data[:,2].astype(np.float64) - imu_data[0,2]).double()

                            # elif (date_dir == '220311'):
                            acc_imu_temp = torch.Tensor(imu_data[:,14:17].astype(np.float64)).double()
                            w_imu_temp = torch.Tensor(imu_data[:,10:13].astype(np.float64)).double()
                            t_imu = torch.Tensor(imu_data[:,2].astype(np.float64) - imu_data[0,2] + imu_data[:,3].astype(np.float64)/1e9 - imu_data[0,3]/1e9).double()
                            

                            acc_imu = mbv(R_imu_from_robot,acc_imu_temp).double()
                            
                            w_imu = mbv(R_imu_from_robot,w_imu_temp).double()
                            device = 'cuda:0'
                            dRot_imu = SO3.exp(self.dt * w_imu.to(device)).double().cpu().detach()
                            Rot_imu = torch.zeros_like(dRot_imu).double()
                            Rot_imu[0] = dRot_imu[0]
                            for i in range(1,dRot_imu.size(0)):
                                Rot_imu[i] = torch.mm(Rot_imu[i-1],dRot_imu[i])

                            quat_imu = SO3.to_quaternion(Rot_imu, ordering='xyzw')

                            rpy_imu = SO3.to_rpy(Rot_imu)

                            # dt_imu = t_imu[1:] - t_imu[:-1]


                        elif (dir == dataset+"-husky_velocity_controller-odom.csv"):
                            wheel_csv = pd.read_csv(path3,sep=",")
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

                            rpy_wheel = SO3.to_rpy(Rot_wheel)

                        elif (dir == dataset+"-odometry-filtered.csv"):
                            ekf_csv = pd.read_csv(path3,sep=",")
                            ekf_data = ekf_csv.to_numpy()

                            p_ekf = torch.Tensor(ekf_data[:,6:9].astype(np.float64)).double()
                            v_ekf = torch.zeros(len(ekf_data),3).double()
                            v_ekf[:,:2] = torch.Tensor(ekf_data[:,14:16].astype(np.float64)).double()

                            t_ekf = torch.Tensor(ekf_data[:,2].astype(np.float64) - ekf_data[0,2] + ekf_data[:,3].astype(np.float64)/1e9 - ekf_data[0,3]/1e9).double()


                            quat_ekf = torch.Tensor(ekf_data[:,9:13].astype(np.float64)).double()

                            Rot_ekf_temp = SO3.from_quaternion(quat_ekf,ordering='xyzw')
                            Rot_ekf_init = Rot_ekf_temp[0].clone().detach()
                            Rot_ekf = mtbm(Rot_ekf_init,Rot_ekf_temp)

                            rpy_ekf = SO3.to_rpy(Rot_ekf)

                            v_loc_ekf = bmtv(Rot_ekf, v_ekf)

            v_wheel_joint_temp = interp_data(t_wheel.numpy(), v_wheel.numpy(), t_imu.numpy())
            v_wheel_joint = torch.Tensor(v_wheel_joint_temp).double()

            features = torch.cat((v_wheel_joint,w_imu,acc_imu),1).float()

            # x,y position: v_gt[:,:2], x,y,z position: v_gt
            targets = torch.cat((v_loc_gt[:,:2], dRot_gt, Rot_gt),1).float()
            # targets = torch.cat((v_loc_gt[:,:2], w_gt),1).float()

            aux = torch.cat((t_gt[:, None], p_gt, rpy_gt, w_gt, t_ekf[:, None], p_ekf, rpy_ekf, v_ekf, w_imu), 1)
            # aux = torch.cat((t_gt[:, None], p_gt, rpy_gt, w_gt), 1)

            mondict = {
            't_gt': t_gt, 'p_gt': p_gt, 'Rot_gt': Rot_gt,'rpy_gt': rpy_gt, 'v_gt': v_gt, 'v_loc_gt': v_loc_gt,
                'quat_gt': quat_gt, 't_imu': t_imu, 'acc_imu': acc_imu, 'w_imu':w_imu, 'Rot_imu': Rot_imu, 
                'rpy_imu': rpy_imu,'quat_imu': quat_imu, 'v_wheel_joint': v_wheel_joint,'features': features, 'targets': targets, 'aux': aux 
                }
            dump(mondict, self.path_data_save, dataset + "_" + self.mode + ".p")
            
            self.features_all.append(features)
            self.targets_all.append(targets)
            self.aux_all.append(aux)

class HUSKYLSTM(Dataset):
    def __init__(self, data, random_shift, shuffle, args):
        super().__init__()

        self.feature_dim = data.feature_dim
        self.target_dim = data.target_dim
        self.step_size = args.step_size
        self.window_size = args.window_size
        self.random_shift = random_shift
        self.shuffle = shuffle

        self.index_map = []

        self.gt_t, self.gt_rpy, self.gt_pos, self.gt_w = [], [], [], []
        self.ekf_t, self.ekf_rpy, self.ekf_pos, self.ekf_v_loc, self.imu_w = [], [], [], [], []

        self.features = data.features_all
        self.targets = data.targets_all
        self.aux = data.aux_all
        


        for i in range(len(data.data_list)):
            self.gt_t.append(self.aux[i][:, 0])
            self.gt_pos.append(self.aux[i][:, 1:4])
            self.gt_rpy.append(self.aux[i][:, 4:7])
            self.gt_w.append(self.aux[i][:,7:10])
            self.ekf_t.append(self.aux[i][:,10])
            self.ekf_pos.append(self.aux[i][:,11:14])
            self.ekf_rpy.append(self.aux[i][:,14:17])
            self.ekf_v_loc.append(self.aux[i][:,17:20])
            self.imu_w.append(self.aux[i][:,20:23])

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

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id - self.window_size:frame_id]

        return feat, targ, seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].unsqueeze(0), self.targets[i]

class HUSKYResNet(Dataset):
    def __init__(self, data, random_shift, shuffle, args):
        super().__init__()

        self.feature_dim = data.feature_dim
        self.target_dim = data.target_dim
        self.step_size = args.step_size
        self.window_size = args.window_size
        self.random_shift = random_shift
        self.shuffle = shuffle

        self.index_map = []

        self.gt_t, self.gt_rpy, self.gt_pos, self.gt_w = [], [], [], []
        self.ekf_t, self.ekf_rpy, self.ekf_pos, self.ekf_v_loc = [], [], [], []

        self.features = data.features_all
        self.targets = data.targets_all
        self.aux = data.aux_all

        for i in range(len(data.data_list)):
            self.gt_t.append(self.aux[i][:, 0])
            self.gt_pos.append(self.aux[i][:, 1:4])
            self.gt_rpy.append(self.aux[i][:, 4:7])
            self.gt_w.append(self.aux[i][:,7:10])
            self.ekf_t.append(self.aux[i][:,10])
            self.ekf_pos.append(self.aux[i][:,11:14])
            self.ekf_rpy.append(self.aux[i][:,14:17])
            self.ekf_v_loc.append(self.aux[i][:,17:20])

            self.index_map += [[i, j] for j in range(self.window_size + self.random_shift, self.targets[i].shape[0], self.step_size)]

        if self.shuffle == True:
            random.shuffle(self.index_map)


    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size : frame_id]
        targ = self.targets[seq_id][frame_id]

        # return feat.astype(np.float32).T, targ, seq_id, frame_id
        return feat.T, targ, seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

class HUSKYCNN(Dataset):
    def __init__(self, data, random_shift, shuffle, args):
        super().__init__()

        self.feature_dim = data.feature_dim
        self.target_dim = data.target_dim
        self.step_size = args.step_size
        self.window_size = args.window_size
        self.random_shift = random_shift
        self.shuffle = shuffle

        self.index_map = []

        self.gt_t, self.gt_rpy, self.gt_pos, self.gt_w = [], [], [], []
        self.ekf_t, self.ekf_rpy, self.ekf_pos, self.ekf_v_loc = [], [], [], []

        self.features = data.features_all
        self.targets = data.targets_all
        self.aux = data.aux_all
        


        for i in range(len(data.data_list)):
            self.features[i] = self.features[i]
            self.targets[i] = self.targets[i]
            self.gt_t.append(self.aux[i][:, 0])
            self.gt_pos.append(self.aux[i][:, 1:4])
            self.gt_rpy.append(self.aux[i][:, 4:7])
            self.gt_w.append(self.aux[i][:,7:10])
            self.ekf_t.append(self.aux[i][:,10])
            self.ekf_pos.append(self.aux[i][:,11:14])
            self.ekf_rpy.append(self.aux[i][:,14:17])
            self.ekf_v_loc.append(self.aux[i][:,17:20])

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

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id - self.window_size:frame_id]

        return feat, targ, seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].unsqueeze(0), self.targets[i]

