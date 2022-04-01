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


from model_temporal import LSTMSeqNetwork, BilinearLSTMSeqNetwork, TCNSeqNetwork
from utils import load_config, MSEAverageMeter
from dataset_RONIN import *
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error

'''
Temporal models with loss functions in global coordinate frame
Configurations
    - Model types 
        TCN - type=tcn
        LSTM_simple - type=lstm, lstm_bilinear        
'''

torch.multiprocessing.set_sharing_strategy('file_system')
_nano_to_sec = 1e09
# _input_channel, _output_channel = 6, 2
_input_channel, _output_channel = 8, 2
# device = 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GlobalPosLoss(torch.nn.Module):
    def __init__(self, mode='full', history=None):
        """
        Calculate position loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalPosLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        assert mode in ['full', 'part']
        self.mode = mode
        if self.mode == 'part':
            assert history is not None
            self.history = history
        elif self.mode == 'full':
            self.history = 1

    def forward(self, pred, targ):
        gt_pos = torch.cumsum(targ[:, 1:, ], 1) #targ.shape: [72,400,2], targ.[:,1:,].shape: [72,399,2]
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)
        if self.mode == 'part':
            gt_pos = gt_pos[:, self.history:, :] - gt_pos[:, :-self.history, :]
            pred_pos = pred_pos[:, self.history:, :] - pred_pos[:, :-self.history, :]
        loss = self.mse_loss(pred_pos, gt_pos)
        return torch.mean(loss)


def write_config(args, **kwargs):
    if args.path_results:
        with open(osp.join(args.path_results, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "pytorch_global_position"
            if kwargs:
                values['kwargs'] = kwargs
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

    data = HuskyRONINData(path_data_base, path_data_save, data_list, mode, args.read_from_data)  
    dataset = HUSKYLSTM(data, random_shift, shuffle, args)

    return dataset

def get_model(args, **kwargs):
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')

    if args.type == 'tcn':
        network = TCNSeqNetwork(_input_channel, _output_channel, args.kernel_size,
                                layer_channels=args.channels, **config)
        print("TCN Network. Receptive field: {} ".format(network.get_receptive_field()))
    elif args.type == 'lstm_bi':
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


def get_loss_function(history, args, **kwargs):
    if args.type == 'tcn':
        config = {'mode': 'part',
                  'history': history}
    else:
        config = {'mode': 'full'}

    criterion = GlobalPosLoss(**config)
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


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset(args.path_data_base, args.path_data_save, args.train_list, args.mode, args)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
    #                           drop_last=True)
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
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    if args.path_results:
        if not osp.isdir(args.path_results):
            os.makedirs(args.path_results)
        if not osp.isdir(osp.join(args.path_results, 'checkpoints')):
            os.makedirs(osp.join(args.path_results, 'checkpoints'))
        if not osp.isdir(osp.join(args.path_results, 'logs')):
            os.makedirs(osp.join(args.path_results, 'logs'))
        # copyfile(args.train_list, osp.join(args.path_results, "train_list"))
        # if args.val_list is not None:
        #     copyfile(args.val_list, osp.join(args.path_results, "validation_list"))
        write_config(args, **kwargs)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_model(args, **kwargs).to(device)
    history = network.get_receptive_field() if args.type == 'tcn' else args.window_size // 2
    criterion = get_loss_function(history, args, **kwargs)

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.75, verbose=True, eps=1e-12)
    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', False)

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
            checkpoints = torch.load(args.continue_from, map_location={model_data['device']: args.device})

        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    if kwargs.get('force_lr', False):
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
            train_vel = MSEAverageMeter(3, [2], _output_channel)
            train_loss = 0
            start_t = time.time()

            for bid, batch in enumerate(train_loader):
                feat, targ, _, _ = batch
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                predicted = network(feat)
                train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                loss = criterion(predicted, targ)
                train_loss += loss.cpu().detach().numpy()
                loss.backward()
                optimizer.step()
                step += 1

            train_errs[epoch] = train_loss / train_mini_batches
            end_t = time.time()
            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss: {}, vel_loss {}/{:.6f}'.format(
                    epoch, end_t - start_t, train_errs[epoch], train_vel.get_channel_avg(), train_vel.get_total_avg()))
            log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch],
                                     *train_vel.get_channel_avg())

            saved_model = False
            if val_loader:
                network.eval()
                val_vel = MSEAverageMeter(3, [2], _output_channel)
                val_loss = 0
                for bid, batch in enumerate(val_loader):
                    feat, targ, _, _ = batch
                    feat, targ = feat.to(device), targ.to(device)
                    optimizer.zero_grad()
                    pred = network(feat)
                    val_vel.add(pred.cpu().detach().numpy(), targ.cpu().detach().numpy())
                    val_loss += criterion(pred, targ).cpu().detach().numpy()
                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss, *val_vel.get_channel_avg())
                if not quiet_mode:
                    print('Validation loss: {} vel_loss: {}/{:.6f}'.format(val_loss, val_vel.get_channel_avg(),
                                                                           val_vel.get_total_avg()))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.path_results:
                        model_path = osp.join(args.path_results, 'checkpoints', 'checkpoint_lstm_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_errs[epoch],
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if use_scheduler:
                    scheduler.step(val_loss)

            if args.path_results and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.path_results, 'checkpoints', 'icheckpoint_lstm_%d.pt' % epoch)
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
        model_path = osp.join(args.path_results, 'checkpoints', 'checkpoint_lstm_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)


def recon_traj_with_preds_global(dataset, preds, ind=None, seq_id=0, type='preds', **kwargs):
    ind = ind if ind is not None else np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)

    if type == 'gt':
        pos = dataset.gt_pos[seq_id][:, :2]
    else:
        ts = dataset.ts[seq_id]
        # Compute the global velocity from local velocity.
        dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
        pos = preds * dts
        pos[0, :] = dataset.gt_pos[seq_id][0, :2]
        pos = np.cumsum(pos, axis=0)
    veloc = preds
    ori = dataset.orientations[seq_id]

    return pos, veloc, ori


def test(args, **kwargs):
    global device, _output_channel
    import matplotlib.pyplot as plt

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # if args.test_path is not None:
    #     if args.test_path[-1] == '/':
    #         args.test_path = args.test_path[:-1]
    #     root_dir = osp.split(args.test_path)[0]
    #     test_data_list = [osp.split(args.test_path)[1]]
    # elif args.test_list is not None:
    #     root_dir = args.data_dir if args.data_dir else osp.split(args.test_list)[0]
    #     with open(args.test_list) as f:
    #         test_data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    # else:
    #     raise ValueError('Either test_path or test_list must be specified.')

    # Load the first sequence to update the input and output size
    _ = get_dataset(args.path_data_base, args.path_data_save, [args.test_list[0]], args.mode, args)

    if args.path_results and not osp.exists(args.path_results):
        os.makedirs(args.path_results)

    with open(osp.join(str(Path(args.model_path).parents[1]), 'config.json'), 'r') as f:
        model_data = json.load(f)

    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        # checkpoint = torch.load(args.model_path, map_location={model_data['device']: args.device})
        checkpoint = torch.load(args.model_path)

    network = get_model(args, **kwargs)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    log_file = None
    if args.test_list and args.path_results:
        # log_file = osp.join(args.path_results, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
        log_file = osp.join(args.path_results, args.type + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.model_path + '\n')
            f.write('Seq traj_len velocity ate rte\n')

    losses_vel = MSEAverageMeter(2, [1], _output_channel)
    ate_all, rte_all = [], []
    pred_per_min = 100 * 60

    seq_dataset = get_dataset(args.path_data_base, args.path_data_save, args.test_list, args.mode, args)

    for idx, data in enumerate(args.test_list):
        # assert data == osp.split(seq_dataset.data_path[idx])[1]

        feat, vel = seq_dataset.get_test_seq(idx)
        feat = torch.Tensor(feat).to(device)
        preds = np.squeeze(network(feat).cpu().detach().numpy())[-vel.shape[0]:, :_output_channel]

        ind = np.arange(vel.shape[0])
        vel_losses = np.mean((vel - preds) ** 2, axis=0)
        losses_vel.add(vel, preds)

        print('Reconstructing trajectory')
        pos_pred, gv_pred, _ = recon_traj_with_preds_global(seq_dataset, preds, ind=ind, type='pred', seq_id=idx)
        pos_gt, gv_gt, _ = recon_traj_with_preds_global(seq_dataset, vel, ind=ind, type='gt', seq_id=idx)

        if args.path_results is not None and osp.isdir(args.path_results):
            np.save(osp.join(args.path_results, '{}_{}.npy'.format(data, args.type)),
                    np.concatenate([pos_pred, pos_gt], axis=1))

        ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
        if pos_pred.shape[0] < pred_per_min:
            ratio = pred_per_min / pos_pred.shape[0]
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
        else:
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
        ate_all.append(ate)
        rte_all.append(rte)

        print('Sequence {}, Velocity loss {} / {}, ATE: {}, RTE:{}'.format(data, vel_losses, np.mean(vel_losses), ate,
                                                                           rte))
        log_line = format_string(data, np.mean(vel_losses), ate, rte)

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
            # plt.axis('equal')
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
            ax3.axis('equal')

            ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position of LSTM")

            ax3.grid()

            ax3.legend(['gt', 'Predicted'])

            if args.show_plot:
                plt.show()

            if args.path_results is not None and osp.isdir(args.path_results):
                plt.savefig(osp.join(args.path_results, '{}_{}.png'.format(data, args.type)))

        if log_file is not None:
            with open(log_file, 'a') as f:
                log_line += '\n'
                f.write(log_line)

        plt.close('all')

    ate_all = np.array(ate_all)
    rte_all = np.array(rte_all)

    measure = format_string('ATE', 'RTE', sep='\t')
    values = format_string(np.mean(ate_all), np.mean(rte_all), sep='\t')
    print(measure, '\n', values)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(measure + '\n')
            f.write(values)

class HuskyLSTMArgs():
    # common
    type = 'lstm_bi' # choices=['tcn','lstm','lstm_bi']
    path_data_base = "../../../Datasets/husky_dataset/211207/"
    path_data_save = "../data"
    path_results = "../results"
    feature_sigma = 0.001
    target_sigma = 0.0
    window_size = 400
    step_size = 100
    batch_size = 72
    num_workers = 1
    mode ='test' # choices=['train', 'test'])
    device = 'cuda:0'
    read_from_data = True
    cpu = False


    # training, cross-validation and test dataset
    train_list = ['square_ccw','circle_cw','ribbon','random_1','inf']
    val_list = ['square_cw', 'circle_ccw']
    test_list = ['random_2']

    # lstm
    layers = 3
    layer_size = 100

    # train
    continue_from = None
    epochs = 1000
    save_interval = 200
    lr = 0.0003

    # test
    model_path = "../results/checkpoints/icheckpoint_lstm_999.pt"
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
