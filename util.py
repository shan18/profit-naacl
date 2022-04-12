import os
import torch
from torch.autograd import Variable


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def prGreen(prt):
    print(f'\033[92m {prt}\033[00m')


def prYellow(prt):
    print(f'\033[93m {prt}\033[00m')


def to_numpy(var):
    return var.cpu().data.numpy() if DEVICE == 'cuda' else var.data.numpy()


def to_tensor(ndarray):
    return torch.tensor(ndarray, dtype=torch.float32).to(DEVICE)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_output_folder(parent_dir, env_name, log_dir=None):
    os.makedirs(parent_dir, exist_ok=True)

    if log_dir is not None and not os.path.exists(os.path.join(parent_dir, log_dir)):
        parent_dir = os.path.join(parent_dir, log_dir)
        os.makedirs(parent_dir, exist_ok=True)
        return parent_dir
    else:
        print('Invalid log dir given, creating a new one with another name.')

    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + f'-run{experiment_id}'
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir