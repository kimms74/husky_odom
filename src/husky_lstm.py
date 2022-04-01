import json
import os
import sys
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model_temporal import LSTMSeqNetwork, BilinearLSTMSeqNetwork
from utils import load_config, MSEAverageMeter
from dataset_HUSKY import *
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error

'''
Temporal models with loss functions in global coordinate frame
Configurations
    - Model types 
        LSTM_simple - type=lstm, lstm_bilinear        
'''

torch.multiprocessing.set_sharing_strategy('file_system')
_nano_to_sec = 1e09
_input_channel, _output_channel = 7, 5 # nonholonomic_velocity, w_imu, acc_imu (1,3,3); 2d_v_loc_gt, ang_vel_gt (2,3)
# _input_channel, _output_channel = 8, 5 # holonomic_velocity, w_imu, acc_imu (2,3,3); 2d_v_loc_gt, ang_vel_gt (2,3)
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TripleLoss(torch.nn.Module):
    def __init__(self, dt):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.huber_loss = torch.nn.HuberLoss(reduction='none')
        self.dt = dt

    def forward(self, pred, targ): # pred: 2d_v_loc, ang_vel (2,3), targ: 2d_v_loc, dRot (2,9)

        ### predicts
        # learning data
        pred_loc_v_temp = pred[:,:,:2].reshape(-1,2).double()
        pred_loc_v = torch.zeros(pred_loc_v_temp.size(0),3).double().to(device)
        pred_loc_v[:,:2] = pred_loc_v_temp
        pred_w = pred[:,:,2:].reshape(-1,3).double()
        
        # #for feat data plot test
        # pred_loc_v_temp = pred[:,:,0].reshape(-1).double()
        # pred_loc_v = torch.zeros(pred_loc_v_temp.size(0),3).double().to(device)
        # pred_loc_v[:,0] = pred_loc_v_temp
        # pred_w = pred[:,:,1:4].reshape(-1,3).double()

        ### targets
        targ_loc_v_temp = targ[:,:,:2].reshape(-1,2).double()
        targ_loc_v = torch.zeros(targ_loc_v_temp.size(0),3).double().to(device)
        targ_loc_v[:,:2] = targ_loc_v_temp
        ## targ dRot_gt
        targ_dRot = targ[:,:,2:11].reshape(-1,3,3).double()
        targ_Rot = targ[:,:,11:20].reshape(-1,3,3).double()
        # targ_w = SO3.log(targ_dRot)/self.dt
        ## targ w_gt
        # targ_w = targ[:,:,2:5].reshape(-1,3).double()

        # targ_Rot = torch.zeros_like(targ_dRot).double().to(device).double()
        # targ_Rot[0] = targ_dRot[0]
        # for i in range(1,targ_dRot.size(0)):
        #     targ_Rot[i] = torch.mm(targ_Rot[i-1].clone(),targ_dRot[i])

        # targ_dp = bmv(targ_Rot,targ_loc_v*self.dt)
        # targ_p = torch.cumsum(targ_dp[:,:2],0)
        
        # 1.local vel loss
        # loc_vel_loss = torch.mean(self.mse_loss(pred_loc_v[:,:2],targ_loc_v[:,:2]))
        # loc_vel_loss = torch.sum(self.mse_loss(pred_loc_v[:,:2],targ_loc_v[:,:2]))
        # loc_vel_loss = torch.sum(self.huber_loss(pred_loc_v[:,:2],targ_loc_v[:,:2]))*10
        pred_dp = bmv(targ_Rot, pred_loc_v*self.dt)
        # pred_p = torch.cumsum(pred_dp[:,:2],0)
        targ_dp = bmv(targ_Rot, targ_loc_v*self.dt)
        # targ_p = torch.cumsum(targ_dp[:,:2],0)
        # pos_loss = torch.sum(self.huber_loss(pred_p[:,:2],targ_p[:,:2]))
        # pos_loss = torch.sum(self.huber_loss(pred_dp[:,:2],targ_dp[:,:2]))*10000
        pos_loss = 0
        pos_N = 4
        for k in range(pos_N):
            pred_dp = pred_dp[::2] + pred_dp[1::2]
            targ_dp = targ_dp[::2] + targ_dp[1::2]

            pos_loss += torch.sum(self.huber_loss(pred_dp[:,:2],targ_dp[:,:2]))*10000

        # 2.rot loss
        # pred_dRot = SO3.exp(self.dt*pred_w).double()
        # pred_Rot = torch.zeros_like(pred_dRot).double().to(device).double()
        # pred_Rot[0] = pred_dRot[0]
        # for i in range(1,pred_Rot.size(0)):
        #     ###################################################################how to fix inplace operation: use clone for parameter!
        #     pred_Rot[i] = torch.mm(pred_Rot[i-1].clone(),pred_dRot[i])

        # error_theta = SO3.log(bmtm(pred_Rot,targ_Rot))

        # theta_loss = 0
        # min_N = 5
        # for k in range(min_N):
        #     pred_dRot = pred_dRot[::2].bmm(pred_dRot[1::2])
        #     targ_dRot = targ_dRot[::2].bmm(targ_dRot[1::2])

        #     error_theta = SO3.log(SO3.dnormalize(bmtm(pred_dRot,targ_dRot)))
        #     # error_theta = SO3.log(bmtm(pred_dRot,targ_dRot))
        #     zero_theta = torch.zeros_like(error_theta).double().to(device)

        #     # theta_loss += torch.sum(self.mse_loss(error_theta, zero_theta))
        #     theta_loss += torch.sum(self.huber_loss(error_theta, zero_theta))*4100

        # min_N = 2
        # for k in range(min_N):
        #     pred_dRot = pred_dRot[::2].bmm(pred_dRot[1::2])
        #     targ_dRot = targ_dRot[::2].bmm(targ_dRot[1::2])

        # error_theta = SO3.log(SO3.dnormalize(bmtm(pred_dRot,targ_dRot)))
        # # error_theta = SO3.log(bmtm(pred_dRot,targ_dRot))
        # zero_theta = torch.zeros_like(error_theta).double().to(device)

        # theta_loss = torch.sum(self.mse_loss(error_theta, zero_theta))*30000        
        # theta_loss = torch.sum(self.mse_loss(pred_w, targ_w))
        # theta_loss = torch.sum(self.huber_loss(pred_w, targ_w))


        # # 3.position loss
        # pred_dp = bmv(pred_Rot,pred_loc_v*self.dt)
        # pred_p = torch.cumsum(pred_dp[:,:2],0)
        # pos_loss = torch.mean(self.mse_loss(pred_p,targ_p))
        # # pos_loss = torch.sum(self.mse_loss(pred_p,targ_p))

        # ##rot test
        # targ_t = np.arange(0,targ_dRot.size(0)*0.01,0.01)
        # pred_t = np.arange(0,pred_dRot.size(0)*0.01,0.01)

        # fig2, ax2 = plt.subplots(sharex=True, figsize=(20, 10))

        # pi_up = np.zeros(len(targ_t))
        # pi_down = np.zeros(len(targ_t))
        # for j in range(len(targ_t)):
        #     pi_up[j] = 3.141592
        #     pi_down[j] = -3.141592
        # targ_ang = SO3.to_rpy(targ_Rot)
        # pred_ang = SO3.to_rpy(pred_Rot)
        # ax2.plot(targ_t, targ_ang[:,0].cpu().detach().numpy())
        # ax2.plot(targ_t, targ_ang[:,1].cpu().detach().numpy())
        # ax2.plot(targ_t, targ_ang[:,2].cpu().detach().numpy())
        # ax2.plot(pred_t, pred_ang[:,0].cpu().detach().numpy())
        # ax2.plot(pred_t, pred_ang[:,1].cpu().detach().numpy())
        # ax2.plot(pred_t, pred_ang[:,2].cpu().detach().numpy())
        # ax2.plot(targ_t, pi_up, 'k',linestyle='--')
        # ax2.plot(targ_t, pi_down, 'k',linestyle='--')        
        # ax2.set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
        #             title="Orientation")
        # ax2.grid()
        # # ax2.legend([r'targ', r'pred'])
        # ax2.legend([r'targ_r',r'targ_p',r'targ_y', r'pred_r', r'pred_p', r'pred_y'])

        # ##loc_vel test
        # fig3, ax3 = plt.subplots(sharex=True, figsize=(20,10))

        # ax3.plot(targ_t, targ_loc_v[:,0].cpu().detach().numpy())
        # ax3.plot(targ_t, targ_loc_v[:,1].cpu().detach().numpy())
        # ax3.plot(pred_t, pred_loc_v[:,0].cpu().detach().numpy())
        # ax3.plot(pred_t, pred_loc_v[:,1].cpu().detach().numpy())
        # ax3.grid()
        # ax3.legend([r'targ_x', r'targ_y', r'pred_x', r'pred_y'])
        # ax3.set(title='Local Velocity')

        # ##pos test
        # fig4, ax4 = plt.subplots(sharex=True, figsize=(20,10))
        # ax4.plot(targ_p[:,0].cpu().detach().numpy(), targ_p[:,1].cpu().detach().numpy())
        # ax4.plot(pred_p[:,0].cpu().detach().numpy(), pred_p[:,1].cpu().detach().numpy())
        # ax3.grid()
        # ax4.legend([r'pred_p',r'targ_p'])
        # ax4.set(title='Position')

        # ## w test
        # fig5, ax5 = plt.subplots(sharex=True, figsize=(20,10))
        # ax5.plot(targ_t, targ_w[:,0].cpu().detach().numpy())
        # ax5.plot(targ_t, targ_w[:,1].cpu().detach().numpy())
        # ax5.plot(targ_t, targ_w[:,2].cpu().detach().numpy())
        # ax5.plot(pred_t, pred_w[:,0].cpu().detach().numpy())
        # ax5.plot(pred_t, pred_w[:,1].cpu().detach().numpy())
        # ax5.plot(pred_t, pred_w[:,2].cpu().detach().numpy())
        # ax5.grid()
        # ax5.legend([r'targ_x',r'targ_y',r'targ_z', r'pred_x', r'pred_y', r'pred_z'])

        # plt.show(block=True)

        # loss = loc_vel_loss + theta_loss
        # loss = loc_vel_loss + pos_loss
        loss = pos_loss
        # loss = loc_vel_loss
        # loss = theta_loss
        # loss = pos_loss + theta_loss
        # loss = loc_vel_loss + theta_loss + pos_loss
        # loss = loc_vel_loss + theta_loss_r + theta_loss_p + theta_loss_y
        # print('vel loss: ',loc_vel_loss)
        # print('theta_loss: ', theta_loss)
        # print('theta_loss: ', theta_loss_r, ' ', theta_loss_p, ' ', theta_loss_y)
        # print('pos_loss: ', pos_loss)
        return loss


def write_config(args):
    if args.path_results:
        with open(osp.join(args.path_results, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "pytorch_global_position"
            json.dump(values, f, sort_keys=True)


def get_dataset(path_data_base, path_data_save, data_list, mode, args):
    input_format, output_format = [0, 3, 6], [0, _output_channel]

    random_shift, shuffle = 0, False

    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False

    dt = 1/args.sensor_frequency

    data = HuskyData(path_data_base, path_data_save, data_list, mode, args.read_from_data, dt)
    dataset = HUSKYLSTM(data, random_shift, shuffle, args)

    return dataset

def get_model(args):
    config = {}
    if args.dropout is not None:
        config['dropout'] = args.dropout

    if args.type == 'lstm_bi':
        print("Bilinear LSTM Network")
        network = BilinearLSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                         lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)
    else:
        print("Simple LSTM Network")
        network = LSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                 lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)

    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    return network

def get_loss_function(args):
    dt = 1/args.sensor_frequency
    criterion = TripleLoss(dt)
    return criterion


def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]


def train(args):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset(args.path_data_base, args.path_data_save, args.train_list, args.mode, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    end_t = time.time()

    print('Training set loaded. Time usage: {:.3f}s'.format(end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset(args.path_data_base, args.path_data_save, args.val_list, 'val', args)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('Validation set loaded')

    # global device
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    if args.path_results:
        if not osp.isdir(args.path_results):
            os.makedirs(args.path_results)
        if not osp.isdir(osp.join(args.path_results, 'checkpoints')):
            os.makedirs(osp.join(args.path_results, 'checkpoints'))
        if not osp.isdir(osp.join(args.path_results, 'logs')):
            os.makedirs(osp.join(args.path_results, 'logs'))

        write_config(args)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_model(args).to(device)
    criterion = get_loss_function(args)

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.75, verbose=True, eps=1e-12)
    quiet_mode = args.quiet
    use_scheduler = args.use_scheduler

    log_file = None
    if args.path_results:
        log_file = osp.join(args.path_results, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.path_results, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        with open(osp.join(str(Path(args.continue_from).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)

        if device.type == 'cpu':
            checkpoints = torch.load(args.continue_from, map_location=lambda storage, location: storage)
        else:
            # checkpoints = torch.load(args.continue_from, map_location={model_data['device']: args.device})
            checkpoints = torch.load(args.continue_from)

        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    if args.force_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)

    print("Starting from epoch {}".format(start_epoch))
    try:
        for epoch in range(start_epoch, args.epochs):
            log_line = ''
            network.train()
            train_loss = 0
            start_t = time.time()

            for bid, batch in enumerate(train_loader):
                feat, targ, _, _ = batch
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                predicted = network(feat)
                loss = criterion(predicted, targ)
                # loss = criterion(feat, targ)
                train_loss += loss.cpu().detach().numpy()
                loss.backward()
                optimizer.step()
                step += 1

            train_errs[epoch] = train_loss / train_mini_batches
            end_t = time.time()
            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss: {}'.format(
                    epoch, end_t - start_t, train_errs[epoch]))
            log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch])

            saved_model = False
            if val_loader:
                network.eval()
                val_loss = 0
                for bid, batch in enumerate(val_loader):
                    feat, targ, _, _ = batch
                    feat, targ = feat.to(device), targ.to(device)
                    optimizer.zero_grad()
                    pred = network(feat)
                    val_loss += criterion(pred, targ).cpu().detach().numpy()
                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss)
                if not quiet_mode:
                    print('Validation loss: {}'.format(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.path_results:
                        model_path = osp.join(args.path_results, 'checkpoints', 'checkpoint_husky_lstm_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_errs[epoch],
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if use_scheduler:
                    scheduler.step(val_loss)

            if args.path_results and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.path_results, 'checkpoints', 'icheckpoint_husky_lstm_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)
            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.path_results:
        model_path = osp.join(args.path_results, 'checkpoints', 'checkpoint_husky_lstm_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)

def recon_traj_with_preds_global(dataset, preds, targs, seq_id, type, args):
    dt = 1/args.sensor_frequency

    if type == 'gt':
        pred_loc_v_temp = targs[:,:2]
        pred_loc_v = torch.zeros(pred_loc_v_temp.size(0),3).float().to(device)
        pred_loc_v[:,:2] = pred_loc_v_temp

        pred_dRot = targs[:,2:11].reshape(-1,3,3).float().to(device)
        w = SO3.log(pred_dRot)/dt
        ori = torch.zeros_like(pred_dRot).float().to(device)
        ori[0] = pred_dRot[0]
        for i in range(1,pred_dRot.size(0)):
            ori[i] = torch.mm(ori[i-1],pred_dRot[i])
        rpy = SO3.to_rpy(ori)
        pred_dp = bmv(ori,pred_loc_v*dt)
        pos = torch.cumsum(pred_dp[:,:2],0)
        veloc = targs[:,:2]

        t = dataset.gt_t[seq_id]


    elif type == 'pred':

        pred_loc_v_temp = preds[:,:2]
        # pred_loc_v_temp = targs[:,:2]
        pred_loc_v = torch.zeros(pred_loc_v_temp.size(0),3).float().to(device)
        pred_loc_v[:,:2] = pred_loc_v_temp

        w = preds[:,2:5]
        # w = dataset.imu_w[seq_id].float().to(device)
        pred_dRot = SO3.exp(dt*w)
        # pred_dRot = targs[:,2:11].reshape(-1,3,3).float().to(device)
        ori = torch.zeros_like(pred_dRot).float().to(device)
        ori[0] = pred_dRot[0]
        for i in range(1,pred_dRot.size(0)):
            ori[i] = torch.mm(ori[i-1],pred_dRot[i])
        rpy = SO3.to_rpy(ori)
        pred_dp = bmv(ori,pred_loc_v*dt)
        pos = torch.cumsum(pred_dp[:,:2],0)
        veloc = preds[:,:2]
        t = dataset.gt_t[seq_id]

    elif type == 'ekf':
        pos = dataset.ekf_pos[seq_id][:, :2]
        rpy = dataset.ekf_rpy[seq_id]
        veloc = dataset.ekf_v_loc[seq_id]
        t = dataset.ekf_t[seq_id]
        # w = torch.zeros_like(preds[:,2:5]).float().to(device)
        w = dataset.imu_w[seq_id]
    
    return pos.cpu().detach().numpy(), rpy.cpu().detach().numpy(), veloc.cpu().detach().numpy(), t.cpu().detach().numpy(), w.cpu().detach().numpy()


def test(args):
    global device, _output_channel
    import matplotlib.pyplot as plt

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load the first sequence to update the input and output size
    _ = get_dataset(args.path_data_base, args.path_data_save, [args.test_list[0]], args.mode, args)

    if args.path_results and not osp.exists(args.path_results):
        os.makedirs(args.path_results)

    with open(osp.join(str(Path(args.model_path).parents[1]), 'config.json'), 'r') as f:
        model_data = json.load(f)

    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path)

    network = get_model(args)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    log_file = None
    if args.test_list and args.path_results:
        log_file = osp.join(args.path_results, args.type + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.model_path + '\n')
            f.write('Seq traj_len velocity ate rte\n')


    ate_all_pred, rte_all_pred = [], []
    ate_all_ekf, rte_all_ekf = [], []
    pred_per_min = args.sensor_frequency * 60

    seq_dataset = get_dataset(args.path_data_base, args.path_data_save, args.test_list, args.mode, args)

    for idx, data in enumerate(args.test_list):
        # assert data == osp.split(seq_dataset.data_path[idx])[1]

        feat, targs = seq_dataset.get_test_seq(idx)
        feat = torch.Tensor(feat).to(device)
        # preds = np.squeeze(network(feat).cpu().detach().numpy())[-vel.shape[0]:, :_output_channel]
        preds = network(feat).squeeze()
        # vel_losses = np.mean((targs - preds) ** 2, axis=0)

        print(data,' Reconstructing trajectory')
        pos_pred, rpy_pred, vel_loc_pred, t_pred, w_pred = recon_traj_with_preds_global(seq_dataset, preds, targs, idx, 'pred', args)
        pos_gt, rpy_gt, vel_loc_gt, t_gt, w_gt = recon_traj_with_preds_global(seq_dataset, preds, targs, idx, 'gt', args)
        pos_ekf, rpy_ekf, vel_loc_ekf, t_ekf, w_ekf = recon_traj_with_preds_global(seq_dataset, preds, targs, idx, 'ekf', args)

        if args.path_results is not None and osp.isdir(args.path_results):
            np.save(osp.join(args.path_results, '{}_{}.npy'.format(data, args.type)),
                    np.concatenate([pos_pred, pos_gt], axis=1))

        ate_pred = compute_absolute_trajectory_error(pos_pred, pos_gt)
        ate_ekf = compute_absolute_trajectory_error(pos_ekf, pos_gt)
        if pos_pred.shape[0] < pred_per_min:
            ratio = pred_per_min / pos_pred.shape[0]
            rte_pred = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
            rte_ekf = compute_relative_trajectory_error(pos_ekf, pos_gt, delta=pos_ekf.shape[0] - 1) * ratio
        else:
            rte_pred = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
            rte_ekf = compute_relative_trajectory_error(pos_ekf, pos_gt, delta=pred_per_min)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

        ate_all_pred.append(ate_pred)
        rte_all_pred.append(rte_pred)
        ate_all_ekf.append(ate_ekf)
        rte_all_ekf.append(rte_ekf)

        # print('Sequence {}, Velocity loss {} / {}, ATE: {}, RTE:{}'.format(data, vel_losses, np.mean(vel_losses), ate,
        #                                                                    rte))
        # log_line = format_string(data, np.mean(vel_losses), ate, rte)

        print('Sequence {}, ATE_pred: {}, RTE_pred:{}'.format(data, ate_pred, rte_pred))
        print('Sequence {}, ATE_ekf: {}, RTE_ekf:{}'.format(data, ate_ekf, rte_ekf))
        log_line = format_string(data, ate_pred, rte_pred)

        if not args.fast_test:
            kp = preds.shape[1]
            if kp == 2:
                targ_names = ['vx', 'vy']
            elif kp == 3:
                targ_names = ['vx', 'vy', 'vz']

            # plt.figure('{}'.format(data), figsize=(16, 9))
            # plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
            # plt.plot(pos_gt[:, 0], pos_gt[:, 1])
            # plt.plot(pos_pred[:, 0], pos_pred[:, 1])
            # plt.title(data)
            # plt.legend(['Ground truth', 'Predicted'])
            # plt.subplot2grid((kp, 2), (kp - 1, 0))
            # plt.plot(pos_cum_error)
            # plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
            # for i in range(kp):
            #     plt.subplot2grid((kp, 2), (i, 1))
            #     plt.plot(ind, vel[:, i])
            #     plt.plot(ind, preds[:, i])
            #     plt.legend(['Ground truth', 'Predicted'])
            #     plt.title('{}, error: {:.6f}'.format(targ_names[i], vel_losses[i]))
            # plt.tight_layout()

            # fig3, ax3 = plt.subplots(figsize=(20, 10))
            # ax3.plot(pos_gt[:, 0], pos_gt[:, 1])
            # ax3.plot(pos_pred[:, 0], pos_pred[:, 1])
            # ax3.plot(pos_ekf[:,0], pos_ekf[:,1])

            # ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position of husky")
            # ax3.axis('equal')

            # ax3.grid()

            # ax3.legend(['gt', 'Predicted', 'ekf'])

            fig4, ax4 = plt.subplots(figsize=(20, 10))
            # ax4.plot(t_gt, rpy_gt[:, 0])
            # ax4.plot(t_pred, rpy_pred[:, 0])
            # ax4.plot(t_gt, rpy_gt[:, 1])
            # ax4.plot(t_pred, rpy_pred[:, 1])            
            ax4.plot(t_gt, rpy_gt[:, 2])
            ax4.plot(t_pred, rpy_pred[:, 2])
            ax4.plot(t_ekf, rpy_ekf[:, 2])
            ax4.set(xlabel=r'time (sec)', ylabel=r'$yaw (rad)', title="orientation of husky")

            ax4.grid()

            # ax4.legend(['gt_r', 'Predicted_r','gt_p', 'Predicted_p','gt_y', 'Predicted_y'])
            ax4.legend(['gt_yaw', 'Predicted_yaw','ekf_yaw'])

            # fig5, ax5 = plt.subplots(figsize=(20, 10))
            # ax5.plot(t_gt, vel_loc_gt[:, :2])
            # ax5.plot(t_pred, vel_loc_pred[:, :2])
            # ax5.plot(t_ekf, vel_loc_ekf[:, :2])

            # ax5.set(xlabel=r'time (sec)', ylabel=r'local velocity (m/s)', title="velocity of husky")

            # ax5.grid()

            # ax5.legend(['gt_x', 'gt_y', 'Predicted_x', 'Predicted_y','ekf_x','ekf_y'])


            # fig6, ax6 = plt.subplots(figsize=(20,10))
            # # ax6.plot(t_gt, w_gt[:,0])
            # # ax6.plot(t_gt, w_gt[:,1])
            # ax6.plot(t_gt, w_gt[:,2])
            # # ax6.plot(t_pred, w_pred[:,0])
            # # ax6.plot(t_pred, w_pred[:,1])
            # ax6.plot(t_pred, w_pred[:,2])
            # ax6.plot(t_ekf, w_ekf[:,2])

            # ax6.set(xlabel=r'time (sec)', ylabel=r'angular velocity (rad/s)', title="angular velocity of husky")

            # ax6.grid()

            # ax6.legend(['gt_x', 'gt_y', 'gt_z', 'Predicted_x', 'Predicted_y', 'Predicted_z'])
            # ax6.legend(['gt_z', 'Predicted_z', 'ekf_z'])

            if args.show_plot:
                plt.show()

            if args.path_results is not None and osp.isdir(args.path_results):
                plt.savefig(osp.join(args.path_results, '{}_{}.png'.format(data, args.type)))

        if log_file is not None:
            with open(log_file, 'a') as f:
                log_line += '\n'
                f.write(log_line)

        plt.close('all')

    ate_all_pred = np.array(ate_all_pred)
    rte_all_pred = np.array(rte_all_pred)
    ate_all_ekf = np.array(ate_all_ekf)
    rte_all_ekf = np.array(rte_all_ekf)


    measure_pred = format_string('ATE_pred', 'RTE_pred', sep='\t')
    values_pred = format_string(np.mean(ate_all_pred), np.mean(rte_all_pred), sep='\t')
    print(measure_pred, '\n', values_pred)

    measure_ekf = format_string('ATE_ekf', 'RTE_ekf', sep='\t')
    values_ekf = format_string(np.mean(ate_all_ekf), np.mean(rte_all_ekf), sep='\t')
    print(measure_ekf, '\n', values_ekf)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(measure_pred + '\n')
            f.write(values_pred)

class HuskyLSTMArgs():
    # common
    type = 'lstm_bi' # choices=['lstm','lstm_bi']
    path_data_base = "../../../Datasets/husky_dataset/"
    path_data_save = "../data"
    path_results = "../results"
    feature_sigma = 0.001
    target_sigma = 0.0

    window_size = 600 #일단 600일때 성능이 좋았음!!!!!!!!!!!!!!!!!!!!!!!!!!
    # window_size = 400
    # window_size = 300
    # window_size = 240
    # window_size = 208
    # window_size = 200
    # window_size = 100

    step_size = 150
    # step_size = 100
    # step_size = 75
    # step_size = 60
    # step_size = 52
    # step_size = 50
    # step_size = 25

    batch_size = 72
    # batch_size = 36
    num_workers = 1
    mode ='test' # choices=['train', 'test'])
    device = 'cuda:0'
    read_from_data = True
    cpu = False
    sensor_frequency = 100

    # training, cross-validation and test dataset
    # train_list = ['move1', 'move2', 'move3', 'move4', 'move5', 'move6', 'move7', 'move8', 'move9', 'origin1', 'origin2', 'origin3', 'origin4']
    train_list = ['move1', 'move2', 'move3', 'move4', 'move5', 'move8', 'move10', 'move14', 'move15', 'move16', 'move17', 'move18', 'move19', 'move20', 'move21', 'origin3', 'origin4', 'origin7', 'origin8']
    # train_list = ['move2', 'move3', 'move4', 'move5', 'move6', 'move7', 'move9', 'move11', 'move12', 'move13', 'move14', 'move15', 'move16', 'move17', 'move18', 'move21', 'origin1', 'origin2']
    # train_list = ['move3', 'move4', 'move5', 'move6', 'move7', 'move8', 'move9', 'move10', 'move13', 'move14', 'move15', 'move16', 'move17', 'move18', 'move19', 'move20', 'move21', 'origin3', 'origin4', 'origin7', 'origin8']
    # train_list = ['move1', 'move2', 'move3', 'move4', 'move5', 'move6', 'move7', 'move10', 'move11', 'move12', 'move13', 'move14', 'move15', 'move16', 'move17', 'move18', 'move19', 'move20', 'move21', 'origin2', 'origin3', 'origin4', 'origin7', 'origin8']
    # train_list = ['square_cw', 'square_ccw', 'circle_cw', 'circle_ccw', 'ribbon', 'inf', 'move1', 'move2', 'move3', 'move4', 'move5', 'move6', 'move7', 'move8', 'origin1', 'origin2', 'origin3', 'origin4']
    # val_list = ['origin1', 'move7', 'move8']
    val_list = ['move7', 'move6', 'move11']
    # val_list = ['random_1', 'random_2']
    # test_list = ['origin4', 'move9']
    # test_list = ['move12', 'move13', 'origin1', 'origin2', 'move7', 'move6', 'move11']
    test_list = ['move12', 'move13', 'origin1', 'origin2']
    # test_list = ['origin3']
    # test_list = ['move6']

    # lstm
    layers = 3
    layer_size = 100

    # train
    # continue_from = "../results/checkpoints/icheckpoint_husky_lstm_599.pt"
    # continue_from = "../results/checkpoints/checkpoint_husky_lstm_227.pt"
    continue_from = None
    epochs = 2000
    save_interval = 200
    lr = 0.0003
    quiet = False
    use_scheduler = True
    # use_scheduler = False
    force_lr = False
    dropout = 0.1
    # dropout = None

    # test
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_556.pt" #only dp 600 + origin: 4100 ### i can write paper!!!!!!!!!!!!!!!!!!!!!!!!!
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_356.pt" #only dp 600 + origin: 2000 n:4
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_798.pt" #only dp 600 + origin: 2000 n:4  loss:37.1
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_1758.pt" #only dp 600 + origin: 2000 n:4  lr rate 15
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_849.pt" #only dp 600 + origin: 2000 n:4  lr rate 50 1500 2000
    model_path = "../results/checkpoints/checkpoint_husky_lstm_835.pt" #only dp 600 + origin: 2000 n:4  lr rate 50 1500 1500
    model_path = "../results/checkpoints/checkpoint_husky_lstm_957.pt" #only dRot 600 + origin: 2000 n:4  loss:4.34
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_1943.pt" #only dp 600 + origin: 2000 n:4  lr rate 50 1500 1000
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_1299.pt" #only dp 600 + origin: 2000 n:4  lr rate 50 1500 500
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_330.pt" #only dRot 600 n:5 2000 
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_237.pt" #only dRot 600 n:5 2000 loss: 30.3
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_229.pt" #only dRot 400 n:4 loss:8.1
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_342.pt" #only p 600 + origin n:4    loss:362
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_412.pt" #only p 600 + origin n:3    loss:348 when reduce lerarning rate is very important!!!
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_298.pt" #only dp 600 loss:                 if too fast, then you gonna local minima!
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_376.pt" #only dp 600 loss: 30.9
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_730.pt" #only dp 600 loss: 30.5
    # model_path = "../results/checkpoints/checkpoint_husky_lstm_303.pt" #only dp 400   loss:21.0        
    fast_test = False
    show_plot = True

    '''
    Extra arguments
    Set True: use_scheduler, 
              quite (no output on stdout), 
              force_lr (force lr when a model is loaded from continue_from)
    float:  dropout, 
            max_ori_error (err. threshold for priority grv in degrees)
            max_velocity_norm (filter outliers in training) 
    '''

if __name__ == '__main__':

    args = HuskyLSTMArgs()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("Model path required")
        args.batch_size = 1
        test(args)