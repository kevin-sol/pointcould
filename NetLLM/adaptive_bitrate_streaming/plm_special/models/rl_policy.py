import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
    

INF = 1e5


class OfflineRLPolicy(nn.Module):
    """
    离线强化学习策略网络。
    用于自适应比特率流媒体的策略网络,结合了预训练语言模型和状态编码器。
    """
    def __init__(
            self,
            state_feature_dim,  # 状态特征维度
            bitrate_levels,     # 比特率等级数
            state_encoder,      # 状态编码器
            plm,               # 预训练语言模型
            plm_embed_size,    # PLM嵌入维度
            max_length=None,   # 最大序列长度
            max_ep_len=100,    # 最大回合长度
            device='cuda' if torch.cuda.is_available() else 'cpu',  # 运行设备
            device_out = None,  # 输出设备
            residual = False,   # 是否使用残差连接
            conv_size = 4,      # 卷积核大小
            which_layer = -1,   # 提前停止层数
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.bitrate_levels = bitrate_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        # 回报嵌入
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        # 动作嵌入
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        # 状态特征嵌入(6个不同的状态分量)
        self.embed_state1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state2 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_state3 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state4 = nn.Linear(state_feature_dim * (6 - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state5 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state6 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    

        # 层归一化
        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        # =========== multimodal encoder (end) ===========
    
        # 动作头部网络,用于预测比特率
        # the so-called network head, used for predicting bitrate
        self.action_head = nn.Linear(plm_embed_size, bitrate_levels).to(device)  

        self.device = device
        self.device_out = device_out

        # the following are used for evaluation
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer
        # 除PLM外的所有模块,用于保存和加载
        self.modules_except_plm = nn.ModuleList([  
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln, 
            self.embed_state1, self.embed_state2, self.embed_state3, self.embed_state4, self.embed_state5,
            self.embed_state6, self.action_head
        ])

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        前向传播函数,用于训练。
        Args:
            states: 状态序列
            actions: 动作序列
            returns: 回报序列
            timesteps: 时间步序列
            attention_mask: 注意力掩码
        Returns:
            action_pred: 预测的动作
        """
        assert actions.shape[0] == 1, '批量大小应为1以避免CUDA内存溢出'

        # Step 1: process actions, returns and timesteps first as they are simple
        actions = actions.to(self.device)  # shape: (1, seq_len, 1)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)

        # 1.1 embed action, return, timestep
        action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

        # 1.2 time embeddings are treated similar to positional embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Step 2: process states, turn them into embeddings.
        states = states.to(self.device)  # shape: (1, seq_len, 6, 6)
        states_features = self.state_encoder(states)
        states_embeddings1 = self.embed_state1(states_features[0]) + time_embeddings
        states_embeddings2 = self.embed_state2(states_features[1]) + time_embeddings
        states_embeddings3 = self.embed_state3(states_features[2]) + time_embeddings
        states_embeddings4 = self.embed_state4(states_features[3]) + time_embeddings
        states_embeddings5 = self.embed_state5(states_features[4]) + time_embeddings
        states_embeddings6 = self.embed_state6(states_features[5]) + time_embeddings
        
        # Step 3: stack returns, states, actions embeddings.
        # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1])  # record the positions of action embeddings
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((returns_embeddings[0, i:i + 1], states_embeddings1[0, i:i + 1], states_embeddings2[0, i:i + 1], 
                                       states_embeddings3[0, i:i + 1], states_embeddings4[0, i:i + 1], states_embeddings5[0, i:i + 1], 
                                       states_embeddings6[0, i:i + 1], action_embeddings[0, i:i + 1]), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = (i + 1) * (2 + 6)
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        
        # Step 4: feed stacked embeddings into the plm
        # 4.1 create attention mask
        if attention_mask is None:
            # 1表示可以被注意到,0表示不能
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 5: predict actions
        # we need to locate the logits corresponding to the state embeddings
        # simply using `action_embed_positions[i] - 2` will do.
        logits_used = logits[:, action_embed_positions - 2]
        action_pred = self.action_head(logits_used)

        return action_pred

    def sample(self, state, target_return, timestep, **kwargs):
        """
        采样动作函数,用于评估/测试。
        Args:
            state: 当前状态
            target_return: 目标回报
            timestep: 当前时间步
        Returns:
            bitrate: 采样的比特率动作
        """
        # Step 1: stack previous state, action, return features in the dequeue
        prev_stacked_inputs = []
        for i in range(len(self.states_dq)):
            prev_return_embeddings = self.returns_dq[i]
            prev_state_embeddings = self.states_dq[i]
            prev_action_embeddings = self.actions_dq[i]
            prev_stacked_inputs.append(torch.cat((prev_return_embeddings, prev_state_embeddings, prev_action_embeddings), dim=1))
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # Step 2: process target return and timesteps
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)

        return_embeddings = return_embeddings + time_embeddings

        # Step 4: process state
        state = state.to(self.device)
        state_features = self.state_encoder(state)
        state_embeddings1 = self.embed_state1(state_features[0]) + time_embeddings
        state_embeddings2 = self.embed_state2(state_features[1]) + time_embeddings
        state_embeddings3 = self.embed_state3(state_features[2]) + time_embeddings
        state_embeddings4 = self.embed_state4(state_features[3]) + time_embeddings
        state_embeddings5 = self.embed_state5(state_features[4]) + time_embeddings
        state_embeddings6 = self.embed_state6(state_features[5]) + time_embeddings
        state_embeddings = torch.cat([state_embeddings1, state_embeddings2, state_embeddings3, state_embeddings4,
                                      state_embeddings5, state_embeddings6], dim=1)


        # Step 5: stack return, stage and previous embeddings
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)  # mind the order
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)  # mind the order
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization

        # 1表示可以被注意到,0表示不能
        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 6: predict the bitrate for next chunk
        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred)

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        
        # update deques
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings) 
        self.actions_dq.append(action_embeddings)

        return bitrate
    
    def clear_dq(self):
        """
        清空双端队列并重新初始化
        """
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))

    def _sample(self, logits):
        """
        从logits中采样动作
        Args:
            logits: 动作logits
        Returns:
            idx: 采样的动作索引
            lgprob: 采样动作的对数概率
        """
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob
