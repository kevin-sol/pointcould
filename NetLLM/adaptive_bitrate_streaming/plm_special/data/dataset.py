import numpy as np
from torch.utils.data import Dataset


def discount_returns(rewards, gamma, scale):
    """
    计算折扣回报值
    Args:
        rewards: 奖励序列
        gamma: 折扣因子
        scale: 缩放因子
    Returns:
        returns: 折扣后的回报值序列
    """
    returns = [0 for _ in range(len(rewards))]
    returns[-1] = rewards[-1]  # 最后一个回报值等于最后一个奖励
    for i in reversed(range(len(rewards) - 1)):  # 从后向前计算折扣回报
        returns[i] = rewards[i] + gamma * returns[i + 1]
    for i in range(len(returns)):  # 对回报值进行缩放
        returns[i] /= scale  # scale down return
    return returns


class ExperienceDataset(Dataset):
    """
    经验数据集类,用于包装经验池数据
    继承自PyTorch的Dataset类,支持数据加载器功能
    """
    def __init__(self, exp_pool, gamma=1., scale=10, max_length=30, sample_step=None) -> None:
        """
        :param exp_pool: the experience pool
        :param gamma: the reward discounted factor
        :param scale: the factor to scale the return
        :param max_length: the w value in our paper, see the paper for details.
        
        初始化经验数据集
        Args:
            exp_pool: 经验池对象
            gamma: 奖励折扣因子
            scale: 回报值缩放因子 
            max_length: 轨迹最大长度,对应论文中的w值
            sample_step: 采样步长,默认等于max_length
        """
        if sample_step is None:
            sample_step = max_length

        self.exp_pool = exp_pool
        self.exp_pool_size = len(exp_pool)
        self.gamma = gamma
        self.scale = scale
        self.max_length = max_length

        # 初始化存储列表
        self.returns = []     # 存储折扣回报值
        self.timesteps = []   # 存储时间步
        self.rewards = []     # 存储归一化后的奖励

        self.exp_dataset_info = {}  # 存储数据集相关信息

        self._normalize_rewards()    # 对奖励进行归一化
        self._compute_returns()      # 计算折扣回报值
        self.exp_dataset_info.update({
            'max_action': max(self.actions),  # 记录动作的最大值
            'min_action': min(self.actions)   # 记录动作的最小值
        })

        # 生成数据集索引,按照采样步长划分
        self.dataset_indices = list(range(0, self.exp_pool_size - max_length + 1, min(sample_step, max_length)))
    
    def sample_batch(self, batch_size=1, batch_indices=None):
        """
        从经验池中采样一批数据
        Args:
            batch_size: 批量大小,对于CJS任务应设为1
            batch_indices: 指定的批量索引,默认随机采样
        Returns:
            batch_states: 状态批量数据
            batch_actions: 动作批量数据  
            batch_returns: 回报值批量数据
            batch_timesteps: 时间步批量数据
        """
        if batch_indices is None:
            batch_indices = np.random.choice(len(self.dataset_indices), size=batch_size)
        batch_states, batch_actions, batch_returns, batch_timesteps = [], [], [], []
        for i in range(batch_size):
            states, actions, returns, timesteps = self[batch_indices[i]]
            batch_states.append(states)
            batch_actions.append(actions)
            batch_returns.append(returns)
            batch_timesteps.append(timesteps)
        return batch_states, batch_actions, batch_returns, batch_timesteps
    
    @property
    def states(self):
        """获取经验池中的状态序列"""
        return self.exp_pool.states

    @property
    def actions(self):
        """获取经验池中的动作序列"""
        return self.exp_pool.actions
    
    @property
    def dones(self):
        """获取经验池中的终止标志序列"""
        return self.exp_pool.dones
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.dataset_indices)
    
    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        Args:
            index: 数据索引
        Returns:
            states: 状态序列片段
            actions: 动作序列片段
            returns: 回报值序列片段
            timesteps: 时间步序列片段
        """
        start = self.dataset_indices[index]
        end = start + self.max_length
        return self.states[start:end], self.actions[start:end], self.returns[start:end], self.timesteps[start:end]

    def _normalize_rewards(self):
        """
        对奖励进行归一化处理
        将奖励值映射到[0,1]区间
        """
        min_reward, max_reward = min(self.exp_pool.rewards), max(self.exp_pool.rewards)
        rewards = (np.array(self.exp_pool.rewards) - min_reward) / (max_reward - min_reward)
        self.rewards = rewards.tolist()
        self.exp_dataset_info.update({
            'max_reward': max_reward,
            'min_reward': min_reward,
        })

    def _compute_returns(self):
        """
        计算每个时间步的累积折扣回报值
        
        主要步骤:
        1. 遍历每个episode,通过done标志找到episode的起止位置
        2. 对每个episode计算折扣回报值
        3. 记录每个时间步的序号
        4. 更新数据集信息(最大/最小回报值和时间步)
        """
        episode_start = 0  # 当前episode的起始位置
        while episode_start < self.exp_pool_size:
            try:
                # 寻找当前episode的结束位置(done=True的位置)
                episode_end = self.dones.index(True, episode_start) + 1
            except ValueError:
                # 如果找不到done=True,则将剩余部分作为一个episode
                episode_end = self.exp_pool_size
                
            # 计算当前episode的折扣回报值并添加到returns列表
            self.returns.extend(discount_returns(self.rewards[episode_start:episode_end], self.gamma, self.scale))
            # 记录每个时间步的序号
            self.timesteps += list(range(episode_end - episode_start))
            # 更新下一个episode的起始位置
            episode_start = episode_end
            
        # 确保returns和timesteps长度一致
        assert len(self.returns) == len(self.timesteps)
        
        # 更新数据集信息
        self.exp_dataset_info.update({
            # for normalizing rewards/returns
            # 记录回报值的范围,用于归一化
            'max_return': max(self.returns),
            'min_return': min(self.returns),

            # to help determine the maximum size of timesteps embedding
            # 记录时间步的范围,用于确定时间步嵌入的最大尺寸
            'min_timestep': min(self.timesteps),
            'max_timestep': max(self.timesteps),
        })
