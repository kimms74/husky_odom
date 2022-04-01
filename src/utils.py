import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import torch
import yaml
def prepare_data(args, dataset, dataset_name, i, idx_start=None, idx_end=None, to_numpy=False):
    # get data
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)

    # get start instant
    if idx_start is None:
        idx_start = 0
    # get end instant
    if idx_end is None:
        idx_end = t.shape[0]

    t = t[idx_start: idx_end]
    u = u[idx_start: idx_end]
    ang_gt = ang_gt[idx_start: idx_end]
    v_gt = v_gt[idx_start: idx_end]
    p_gt = p_gt[idx_start: idx_end] - p_gt[idx_start]

    if to_numpy:
        t = t.cpu().double().numpy()
        u = u.cpu().double().numpy()
        ang_gt = ang_gt.cpu().double().numpy()
        v_gt = v_gt.cpu().double().numpy()
        p_gt = p_gt.cpu().double().numpy()
    return t, ang_gt, p_gt, v_gt, u


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)



def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix that minimizes the distance between a set of
    registered points.

    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """


    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

def dump(mondict, *_file_name):
    pickle_extension = ".p"
    file_name = os.path.join(*_file_name)
    if not file_name.endswith(pickle_extension):
        file_name += pickle_extension
    with open(file_name, "wb") as file_pi:
        pickle.dump(mondict, file_pi)

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

def interp_data(t_x, x, t):
    x_int = np.zeros((t.shape[0], x.shape[1]))
    for i in range(0, x.shape[1]):
            x_int[:, i] = np.interp(t, t_x, x[:, i])
    return x_int

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

class MSEAverageMeter:
    def __init__(self, ndim, retain_axis, n_values=3):
        """
        Calculate average without overflows
        :param ndim: Number of dimensions
        :param retain_axis: Dimension to get average along
        :param n_values: Number of values along retain_axis
        """
        self.count = 0
        self.average = np.zeros(n_values, dtype=np.float64)
        self.retain_axis = retain_axis
        self.targets = []
        self.predictions = []
        self.axis = tuple(np.setdiff1d(np.arange(0, ndim), retain_axis))

    def add(self, pred, targ):
        self.targets.append(targ)
        self.predictions.append(pred)
        print(np.shape(pred))
        print(np.shape(targ))
        val = np.average((targ - pred) ** 2, axis=self.axis)
        c = np.prod([targ.shape[i] for i in self.axis])
        ct = c + self.count
        self.average = self.average * (self.count / ct) + val * (c / ct)
        self.count = ct

    def get_channel_avg(self):
        return self.average

    def get_total_avg(self):
        return np.average(self.average)

    def get_elements(self, axis):
        return np.concatenate(self.predictions, axis=axis), np.concatenate(self.targets, axis=axis)


def load_config(default_config, args, unknown_args):
    """
    Combine the arguments passed by user with configuration file given by user [and/or] default configuration. Convert extra named arguments to correct format.
    :param default_config: path to file
    :param args: known arguments passed by user
    :param unknown_args: unknown arguments passed by user
    :return: known_arguments, unknown_arguments
    """
    kwargs = {}

    def convert_value(y):
        try:
            return int(y)
        except:
            pass
        try:
            return float(y)
        except:
            pass
        if y == 'True' or y == 'False':
            return y == 'True'
        else:
            return y

    def convert_arrry(x):
        if not x:
            return True
        elif len(x) == 1:
            return x[0]
        return x

    i = 0
    while i < len(unknown_args):
        if unknown_args[i].startswith('--'):
            token = unknown_args[i].lstrip('-')
            options = []
            i += 1
            while i < len(unknown_args) and not unknown_args[i].startswith('--'):
                options.append(convert_value(unknown_args[i]))
                i += 1
            kwargs[token] = convert_arrry(options)

    if 'config' in kwargs:
        args.config = kwargs['config']
        del kwargs['config']
    with open(args.config, 'r') as f:
        config = json.load(f)

    values = vars(args)

    def add_missing_config(dictionary, remove=False):
        for key in values:
            if values[key] in [None, False] and key in dictionary:
                values[key] = dictionary[key]
                if remove:
                    del dictionary[key]

    add_missing_config(kwargs, True)        # specified args listed as unknowns
    add_missing_config(config)              # configuration from file for unspecified variables
    if args.config != default_config:       # default config
        with open(default_config, 'r') as f:
            default_configs = json.load(f)
        add_missing_config(default_configs)

    try:
        if args.channels is not None and type(args.channels) is str:
            args.channels = [int(i) for i in args.channels.split(',')]
    except:
        pass

    if 'kwargs' in config:
        kwargs = {**config['kwargs'], **kwargs}

    return args, kwargs


def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)

def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.mkdir(path)

def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

def bmv(mat, vec):
    """batch matrix vector product"""
    return torch.einsum('bij, bj -> bi', mat, vec)

def bbmv(mat, vec):
    """double batch matrix vector product"""
    return torch.einsum('baij, baj -> bai', mat, vec)

def bmtv(mat, vec):
    """batch matrix transpose vector product"""
    return torch.einsum('bji, bj -> bi', mat, vec)

def bmtm(mat1, mat2):
    """batch matrix transpose matrix product"""
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def bmmt(mat1, mat2):
    """batch matrix matrix transpose product"""
    return torch.einsum("bij, bkj -> bik", mat1, mat2)

def mbv(mat, vec):
    """matrix batch vector product"""
    return torch.einsum("ij, bj -> bi", mat, vec)

def mtbv(mat, vec):
    """matrix transpose batch vector product"""
    return torch.einsum("ji, bj -> bi", mat, vec)

def mbm(mat1, mat2):
    """matrix batch matrix product"""
    return torch.einsum("ij, bjk -> bik", mat1, mat2)

def mtbm(mat1, mat2):
    """matrix transpose batch matrix product"""
    return torch.einsum("ji, bjk -> bik", mat1, mat2)