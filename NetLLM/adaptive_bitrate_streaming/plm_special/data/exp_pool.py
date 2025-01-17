
class ExperiencePool:
    """
    经验池类,用于收集和存储智能体与环境交互的轨迹数据。
    包含状态(state)、动作(action)、奖励(reward)和终止标志(done)等信息。
    """
    def __init__(self):
        # 初始化存储列表
        self.states = []    # 存储环境状态序列
        self.actions = []   # 存储智能体动作序列  
        self.rewards = []   # 存储获得的奖励序列
        self.dones = []     # 存储是否终止的标志序列

    def add(self, state, action, reward, done):
        """
        向经验池中添加一条轨迹数据
        Args:
            state: 环境状态/观测值 
            action: 智能体采取的动作
            reward: 获得的奖励
            done: 当前episode是否结束的标志
        """
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward) 
        self.dones.append(done)

    def __len__(self):
        """
        返回经验池中数据的数量
        Returns:
            经验池中存储的轨迹数据长度
        """
        return len(self.states)

