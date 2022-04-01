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


from utils import load_config, MSEAverageMeter
from dataset_HUSKY import *
from model_resnet1d import *
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error, compute_ate_rte

_input_channel, _output_channel = 7, 5 # nonholonomic_velocity, w_imu, acc_imu (1,3,3); 2d_v_loc_gt, ang_vel_gt (2,3)
# _input_channel, _output_channel = 8, 5 # holonomic_velocity, w_imu, acc_imu (2,3,3); 2d_v_loc_gt, ang_vel_gt (2,3)

_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def write_config(args):
    if args.path_results:
        with open(osp.join(args.path_results, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "pytorch_global_position"
            json.dump(values, f, sort_keys=True)

def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)

def get_dataset(path_data_base, path_data_save, data_list, mode, args):
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
    dataset = HUSKYResNet(data, random_shift, shuffle, args)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim

    return dataset

def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, batch in enumerate(data_loader):
        feat, targ, _, _ = batch
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all

def get_model(arch):
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
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

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch).to(device)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)
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
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    step = 0
    best_val_loss = np.inf

    print("Starting from epoch {}".format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)

    init_train_loss = np.sum((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.sum(init_train_loss))
    print('-------------------------')
    print('Init: sum loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device)
        init_val_loss = np.sum((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.sum(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
    
    try:
        for epoch in range(start_epoch, args.epochs):
            start_t = time.time()
            network.train()
            train_outs, train_targets = [], []
            for batch_id, (feat, targ, _, _) in enumerate(train_loader):
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                pred = network(feat)
                train_outs.append(pred.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                loss = criterion(pred, targ)
                loss = torch.sum(loss)
                loss.backward()
                optimizer.step()
                step += 1
            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.sum((train_outs - train_targets) ** 2, axis=0)

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, sum loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.sum(train_losses)))
            train_losses_all.append(np.sum(train_losses))


            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(network, val_loader, device)
                val_losses = np.sum((val_outs - val_targets) ** 2, axis=0)
                sum_loss = np.sum(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, sum_loss))

                val_losses_all.append(sum_loss)
                if sum_loss < best_val_loss:
                    best_val_loss = sum_loss
                    if args.path_results and osp.isdir(args.path_results):
                        model_path = osp.join(args.path_results, 'checkpoints', 'checkpoint_resnet_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
                if use_scheduler:
                    scheduler.step(sum_loss)

            else:
                if args.path_results is not None and osp.isdir(args.path_results):
                    model_path = osp.join(args.path_results, 'checkpoints', 'checkpoint_resnet_%d.pt' % epoch)
                    torch.save({'model_state_dict': network.state_dict(),
                                'epoch': epoch,
                                'optimizer_state_dict': optimizer.state_dict()}, model_path)
                    print('Best Validation Model saved to ', model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.path_results:
        model_path = osp.join(args.path_results, 'checkpoints', 'checkpoint_resnet_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all

def recon_traj_with_preds_global(dataset, preds, type, args, seq_id=0):
    if type == 'gt':
        # pos = dataset.gt_pos[seq_id][:, :2]
        # rpy = dataset.gt_rpy[seq_id]
        # veloc = torch.Tensor(preds[:,:2])
        # t = dataset.gt_t[seq_id]
        # w = dataset.gt_w[seq_id]

        dt = 1/args.sensor_frequency

        pred_loc_v_temp = torch.Tensor(preds[:,:2])
        pred_loc_v = torch.zeros(pred_loc_v_temp.size(0),3).float().to(device)
        pred_loc_v[:,:2] = pred_loc_v_temp
        w = torch.Tensor(preds[:,2:5]).float().to(device)
        pred_dRot = SO3.exp(dt*w)
        ori = torch.zeros_like(pred_dRot).float().to(device)
        ori[0] = pred_dRot[0]
        for i in range(1,pred_dRot.size(0)):
            ori[i] = torch.mm(ori[i-1],pred_dRot[i])
        rpy = SO3.to_rpy(ori)
        pred_dp = bmv(ori,pred_loc_v*dt)
        pos = torch.cumsum(pred_dp[:,:2],0)
        veloc = torch.Tensor(preds[:,:2])
        t = torch.arange(start=0, end= veloc.size(0)*dt, step=dt)

    elif type == 'pred':
        dt = 1/args.sensor_frequency

        pred_loc_v_temp = torch.Tensor(preds[:,:2])
        pred_loc_v = torch.zeros(pred_loc_v_temp.size(0),3).float().to(device)
        pred_loc_v[:,:2] = pred_loc_v_temp
        w = torch.Tensor(preds[:,2:5]).float().to(device)
        pred_dRot = SO3.exp(dt*w)
        ori = torch.zeros_like(pred_dRot).float().to(device)
        ori[0] = pred_dRot[0]
        for i in range(1,pred_dRot.size(0)):
            ori[i] = torch.mm(ori[i-1],pred_dRot[i])
        rpy = SO3.to_rpy(ori)
        pred_dp = bmv(ori,pred_loc_v*dt)
        pos = torch.cumsum(pred_dp[:,:2],0)
        veloc = torch.Tensor(preds[:,:2])
        # t = dataset.gt_t[seq_id]
        t = torch.arange(start=0, end= veloc.size(0)*dt, step=dt)

    elif type == 'ekf':
        pos = dataset.ekf_pos[seq_id][:, :2]
        rpy = dataset.ekf_rpy[seq_id]
        veloc = dataset.ekf_v_loc[seq_id]
        t = dataset.ekf_t[seq_id]
        w = torch.zeros_like(torch.Tensor(preds[:,2:5])).float().to(device)
    
    return pos.cpu().detach().numpy(), rpy.cpu().detach().numpy(), veloc.cpu().detach().numpy(), t.cpu().detach().numpy(), w.cpu().detach().numpy()


def test(args):
    global device, _output_channel
    import matplotlib.pyplot as plt

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    # Load the first sequence to update the input and output size
    _ = get_dataset(args.path_data_base, args.path_data_save, [args.test_list[0]], args.mode, args)

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch)

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = args.sensor_frequency * 60

    for data in args.test_list:
        seq_dataset = get_dataset(args.path_data_base, args.path_data_save, [data], 'test', args)
        seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)

        targets, preds = run_test(network, seq_loader, device, True)
        losses = np.sum((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        pos_pred, rpy_pred, vel_loc_pred, t_pred, w_pred = recon_traj_with_preds_global(seq_dataset, preds, 'pred', args)
        pos_gt, rpy_gt, vel_loc_gt, t_gt, w_gt = recon_traj_with_preds_global(seq_dataset, targets, 'gt', args)
        traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
        ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

        print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.sum(losses), ate, rte))
        log_line = format_string(data, ate, rte)

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

        fig3, ax3 = plt.subplots(figsize=(20, 10))
        ax3.plot(pos_gt[:, 0], pos_gt[:, 1])
        ax3.plot(pos_pred[:, 0], pos_pred[:, 1])
        # ax3.plot(pos_ekf[:,0], pos_ekf[:,1])

        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position of ROT")
        ax3.axis('equal')

        ax3.grid()

        ax3.legend(['gt', 'Predicted', 'ekf'])

        fig4, ax4 = plt.subplots(figsize=(20, 10))
        # ax4.plot(t_gt, rpy_gt[:, 0])
        # ax4.plot(t_pred, rpy_pred[:, 0])
        # ax4.plot(t_gt, rpy_gt[:, 1])
        # ax4.plot(t_pred, rpy_pred[:, 1])            
        ax4.plot(t_gt, rpy_gt[:, 2])
        ax4.plot(t_pred, rpy_pred[:, 2])
        ax4.set(xlabel=r'time (sec)', ylabel=r'$yaw (rad)', title="orientation of ROT")

        ax4.grid()

        # ax4.legend(['gt_r', 'Predicted_r','gt_p', 'Predicted_p','gt_y', 'Predicted_y'])
        ax4.legend(['gt_yaw', 'Predicted_yaw','ekf_yaw'])

        fig5, ax5 = plt.subplots(figsize=(20, 10))
        ax5.plot(t_gt, vel_loc_gt[:, :2])
        ax5.plot(t_pred, vel_loc_pred[:, :2])
        # ax5.plot(t_ekf, vel_loc_ekf[:, :2])

        ax5.set(xlabel=r'time (sec)', ylabel=r'local velocity (m/s)', title="velocity of ROT")

        ax5.grid()

        ax5.legend(['gt_x', 'gt_y', 'Predicted_x', 'Predicted_y','ekf_x','ekf_y'])


        # fig6, ax6 = plt.subplots(figsize=(20,10))
        # ax6.plot(t_gt, w_gt[:,0])
        # ax6.plot(t_gt, w_gt[:,1])
        # ax6.plot(t_gt, w_gt[:,2])
        # ax6.plot(t_pred, w_pred[:,0])
        # ax6.plot(t_pred, w_pred[:,1])
        # ax6.plot(t_pred, w_pred[:,2])

        # ax6.set(xlabel=r'time (sec)', ylabel=r'angular velocity (rad/s)', title="angular velocity of ROT")

        # ax6.grid()

        # ax6.legend(['gt_x', 'gt_y', 'gt_z', 'Predicted_x', 'Predicted_y', 'Predicted_z'])

        if args.show_plot:
            plt.show()

        if args.path_results is not None and osp.isdir(args.path_results):
            np.save(osp.join(args.path_results, data + '_gsn.npy'),
                    np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
            plt.savefig(osp.join(args.path_results, data + '_gsn.png'))

        plt.close('all')

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    # Export a csv file
    if args.path_results is not None and osp.isdir(args.path_results):
        with open(osp.join(args.path_results, 'losses.csv'), 'w') as f:
            if losses_seq.shape[1] == 2:
                f.write('seq,vx,vy,avg,ate,rte\n')
            else:
                f.write('seq,vx,vy,vz,avg,ate,rte\n')
            for i in range(losses_seq.shape[0]):
                f.write('{},'.format(args.test_list[i]))
                for j in range(losses_seq.shape[1]):
                    f.write('{:.6f},'.format(losses_seq[i][j]))
                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

    print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
        np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
    return losses_avg

class HuskyResnetArgs():
    # common
    path_data_base = "../../../Datasets/husky_dataset/"
    path_data_save = "../data"
    path_results = "../results"
    window_size = 100
    step_size = 10
    batch_size = 128
    num_workers = 1
    mode ='test' # choices=['train', 'test'])
    device = 'cuda:0'
    read_from_data = True
    cpu = False
    sensor_frequency = 100

    # training, cross-validation and test dataset
    train_list = ['move1', 'move2', 'move3', 'move4', 'move5', 'move6', 'move7', 'origin1', 'origin2', 'origin3', 'origin4']
    val_list = ['move8']
    test_list = ['move9']

    # resnet
    arch = 'resnet18'

    # train
    # continue_from = "../results/checkpoints/checkpoint_ROT_844.pt"
    # continue_from = "../results/checkpoints/checkpoint_husky_3576.pt"
    continue_from = None
    epochs = 4000
    lr = 1e-04
    dropout = 0.1
    # dropout = None
    use_scheduler = True

    # test
    model_path = "../results/checkpoints/checkpoint_resnet_1219.pt" 
    show_plot = True

if __name__ == '__main__':

    args = HuskyResnetArgs()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError('Undefined mode')