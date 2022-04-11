import os
import torch
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def prRed(prt):
    print(f'\033[91m {prt}\033[00m')


def prGreen(prt):
    print(f'\033[92m {prt}\033[00m')


def prYellow(prt):
    print(f'\033[93m {prt}\033[00m')


def prLightPurple(prt):
    print(f'\033[94m {prt}\033[00m')


def prPurple(prt):
    print(f'\033[95m {prt}\033[00m')


def prCyan(prt):
    print(f'\033[96m {prt}\033[00m')


def prLightGray(prt):
    print(f'\033[97m {prt}\033[00m')


def prBlack(prt):
    print(f'\033[98m {prt}\033[00m')


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_output_folder(parent_dir, env_name):
    os.makedirs(parent_dir, exist_ok=True)
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