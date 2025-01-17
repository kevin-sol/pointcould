import copy
import os
import torch
import time

from baseline_special.env import Environment
from baseline_special.utils.constants import (
    REBUF_PENALTY, SMOOTH_PENALTY, DEFAULT_QUALITY, S_INFO, S_LEN, BITRATE_LEVELS, BUFFER_NORM_FACTOR,
    M_IN_K, SMOOTH_PENALTY, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP, MAX_VIDEO_BIT_RATE, DEFAULT_QUALITY
)
from plm_special.utils.utils import calc_mean_reward, clear_dir, set_random_seed


def test_on_env(args, model, results_dir, env_settings, target_return, max_ep_num=100, process_reward_fn=None, seed=0):
    """
    在环境中测试模型性能。

    Args:
        args: 参数配置
        model: 要测试的模型
        results_dir: 结果保存目录
        env_settings: 环境设置
        target_return: 目标回报值
        max_ep_num: 最大测试回合数,默认100
        process_reward_fn: 奖励处理函数,默认为None
        seed: 随机种子,默认为0

    Returns:
        test_log: 包含测试结果的字典
    """
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    test_log = {}
    test_start = time.time()
    
    results_log = {}
    with torch.no_grad():
        # 初始化环境
        env = Environment(**env_settings)
        
        # 初始化状态和参数
        time_stamp = 0
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        state = torch.zeros((1, 1, S_INFO, S_LEN), dtype=torch.float32, device=args.device)
        timestep = 0
        target_return_clone = copy.deepcopy(target_return)
        ep_count = 0
        episodes_return, episodes_len = 0, 0
    
        trace_idx = env.trace_idx
        results_log[trace_idx] = []

        set_random_seed(args.seed)

        while True:
            # 从环境获取视频块信息
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)

            time_stamp += delay  # 延迟时间(ms)
            time_stamp += sleep_time  # 休眠时间(ms)
            
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            
            # 计算平滑度
            smoothness = abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            last_bit_rate = bit_rate

            # 记录结果
            results_log[trace_idx].append([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate], buffer_size,
                                           rebuf, video_chunk_size, delay, smoothness, reward])

            # dequeue history record
            state = torch.roll(state, -1, dims=-1)

            # 更新状态信息
            state[..., 0, -1] = VIDEO_BIT_RATE[bit_rate] / MAX_VIDEO_BIT_RATE # 上一个质量
            state[..., 1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 缓冲区大小(10秒)
            state[..., 2, -1] = video_chunk_size / delay / M_IN_K  # 吞吐量(KB/ms)
            state[..., 3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 延迟(10秒)
            state[..., 4, :BITRATE_LEVELS] = torch.as_tensor(next_video_chunk_sizes, device=args.device, dtype=torch.float32) / M_IN_K / M_IN_K  # 下一个块大小(MB)
            state[..., 5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP # 剩余块数

            if timestep > 0:  # 跳过第一个奖励(与pensieve一致)
                reward = process_reward_fn(reward)
                target_return = target_return - reward
                episodes_return += reward
                episodes_len += 1

            # 根据当前状态采样下一个比特率
            bit_rate = model.sample(state, target_return, timestep)
            timestep += 1

            if end_of_video:
                # 视频结束时重置状态
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                torch.zero_(state)
                timestep = 0
                target_return = copy.deepcopy(target_return_clone)

                ep_count += 1
                if ep_count >= max_ep_num:
                    break

                trace_idx = env.trace_idx
                results_log[trace_idx] = []
    
    test_log.update({'time': time.time() - test_start})

    # 将结果写入磁盘
    clear_dir(results_dir)  # 先清空目录
    all_file_names = env_settings['all_file_names']
    for trace_idx, values in results_log.items():
        result_path = os.path.join(results_dir, 'result_sim_abr_{}'.format(all_file_names[trace_idx]))
        with open(result_path, 'w') as result_file:
            for items in values:
                time_stamp, bit_rate, buffer_size, rebuf, video_chunk_size, download_time, smoothness, reward = items
                # 按格式记录:时间戳 比特率 缓冲区大小 重缓冲时间 块大小 下载时间 平滑度 奖励
                result_file.write(str(time_stamp) + '\t' +
                                  str(bit_rate) + '\t' +
                                  str(buffer_size) + '\t' +
                                  str(rebuf) + '\t' +
                                  str(video_chunk_size) + '\t' +
                                  str(download_time) + '\t' +
                                  str(smoothness) + '\t' +
                                  str(reward) + '\n' )
            result_file.close()
    test_log['mean_reward'] = calc_mean_reward(result_files=os.listdir(results_dir), test_dir=results_dir, str='', skip_first_reward=True)
    return test_log
