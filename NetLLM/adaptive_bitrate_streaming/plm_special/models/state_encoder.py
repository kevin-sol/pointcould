"""
Customized state encoder based on Pensieve's encoder.
"""
import torch.nn as nn


class EncoderNetwork(nn.Module):
    """
    The encoder network for encoding each piece of information of the state.
    This design of the network is from Pensieve/Genet.
    """
    def __init__(self, conv_size=4, bitrate_levels=6, embed_dim=128):
        """
        初始化编码器网络。

        Args:
            conv_size: 卷积核大小,默认为4
            bitrate_levels: 比特率等级数,默认为6 
            embed_dim: 嵌入维度,默认为128
        """
        super().__init__()
        self.past_k = conv_size
        self.bitrate_levels = 6
        self.embed_dim = embed_dim
        # 编码上一个比特率
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  
        # 编码当前缓冲区大小
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  
        # 编码过去k个吞吐量
        self.conv3 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  
        # 编码过去k个下载时间
        self.conv4 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  
        # 编码下一个块的大小
        self.conv5 = nn.Sequential(nn.Conv1d(1, embed_dim, bitrate_levels), nn.LeakyReLU(), nn.Flatten())  
        # 编码剩余块数
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())        


    def forward(self, state):
        """
        前向传播函数。

        Args:
            state: 输入状态张量,形状为(batch_size, seq_len, 6, 6)

        Returns:
            features1-6: 6个编码后的特征张量,每个形状为(batch_size, seq_len, embed_dim)
        """
        # 将输入状态重塑为(batch_size x seq_len, 6, 6)
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, 6, 6)
        
        # 提取各个状态分量
        last_bitrate = state[..., 0:1, -1]  # 上一个比特率
        current_buffer_size = state[..., 1:2, -1]  # 当前缓冲区大小
        throughputs = state[..., 2:3, :]  # 吞吐量序列
        download_time = state[..., 3:4, :]  # 下载时间序列
        next_chunk_size = state[..., 4:5, :self.bitrate_levels]  # 下一块大小
        remain_chunks = state[..., 5:6, -1]  # 剩余块数
        
        # 对各个状态分量进行编码
        features1 = self.fc1(last_bitrate).reshape(batch_size, seq_len, -1)
        features2 = self.fc2(current_buffer_size).reshape(batch_size, seq_len, -1)
        features3 = self.conv3(throughputs).reshape(batch_size, seq_len, -1)
        features4 = self.conv4(download_time).reshape(batch_size, seq_len, -1)
        features5 = self.conv5(next_chunk_size).reshape(batch_size, seq_len, -1)
        features6 = self.fc6(remain_chunks).reshape(batch_size, seq_len, -1)
        return features1, features2, features3, features4, features5, features6
