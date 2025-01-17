import os
import random
import numpy as np
from baseline_special.utils.constants import BITRATE_LEVELS
try:  # baselines use a different conda environment without torch, so we need to skip ModuleNotFoundError when runing baselines
    import torch
except ModuleNotFoundError:
    pass


def process_batch(batch, device='cpu'):
    """
    Process batch of data.
    处理数据批次。
    将状态、动作、回报和时间步骤转换为适当的张量格式。

    Args:
        batch: 包含状态、动作、回报和时间步骤的批次数据
        device: 运行设备,默认为CPU

    Returns:
        states: 状态张量
        actions: 动作张量
        returns: 回报张量 
        timesteps: 时间步骤张量
        labels: 标签张量
    """
    states, actions, returns, timesteps = batch

    states = torch.cat(states, dim=0).unsqueeze(0).float().to(device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(1, -1)
    labels = actions.long()
    actions = ((actions + 1) / BITRATE_LEVELS).unsqueeze(2)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1)
    timesteps = torch.as_tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)

    return states, actions, returns, timesteps, labels


def set_random_seed(seed):
    """
    设置随机种子以确保实验可重复性。
    
    Args:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def action2bitrate(action, last_bit_rate):
    """
    Genet special.
    Genet uses this strategy for converting actions to bitrates during testing.
    
    Genet专用的动作到比特率的转换函数。
    在测试期间使用此策略将动作转换为比特率。

    Args:
        action: 选择的动作(0-2)
        last_bit_rate: 上一个比特率

    Returns:
        bit_rate: 转换后的比特率值(0-5)
    """
    # selected_action is 0-2
    # naive step implementation
    if action == 1:
        bit_rate = last_bit_rate
    elif action == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    if bit_rate < 0:
        bit_rate = 0
    if bit_rate > 5:
        bit_rate = 5
    return bit_rate


def calc_mean_reward(result_files, test_dir, str, skip_first_reward=True):
    """
    计算平均奖励。

    Args:
        result_files: 结果文件列表
        test_dir: 测试目录路径
        str: 用于匹配文件的字符串
        skip_first_reward: 是否跳过第一个奖励值,默认为True

    Returns:
        float: 平均奖励值
    """
    matching = [s for s in result_files if str in s]
    reward = []
    count = 0
    for log_file in matching:
        count += 1
        first_line = True
        with open(test_dir + '/' + log_file, 'r') as f:
            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                if first_line:
                    first_line = False
                    if skip_first_reward:
                        continue
                reward.append(float(parse[7]))
    print(count)
    return np.mean(reward)


def clear_dir(directory):
    """
    清空指定目录下的所有文件。

    Args:
        directory: 要清空的目录路径
    """
    file_list = os.listdir(directory)
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)