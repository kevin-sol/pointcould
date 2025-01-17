import numpy as np
import torch
import time

from munch import Munch
from torch.utils.data import DataLoader

from plm_special.utils.utils import process_batch


class Trainer:
    def __init__(self, args, model, optimizer, exp_dataset, loss_fn, device, batch_size=1, grad_accum_steps=1, lr_scheduler=None):
        """
        训练器类的初始化函数。

        Args:
            args: 参数配置
            model: 要训练的模型
            optimizer: 优化器
            exp_dataset: 经验数据集
            loss_fn: 损失函数
            device: 运行设备
            batch_size: 批次大小,默认为1
            grad_accum_steps: 梯度累积步数,默认为1
            lr_scheduler: 学习率调度器,默认为None
        """
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.exp_dataset = exp_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        
        self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
        self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)

    def train_epoch(self, report_loss_per_steps=100):
        """
        训练一个epoch。

        Args:
            report_loss_per_steps: 每多少步报告一次损失,默认为100

        Returns:
            logs: 包含训练信息的字典
            train_losses: 训练损失列表
        """
        train_losses = []
        logs = dict()

        train_start = time.time()
        dataset_size = len(self.dataloader)

        self.model.train()
        for step, batch in enumerate(self.dataloader):
            # 执行训练步骤
            train_loss = self.train_step(batch)
            train_losses.append(train_loss.item())

            # perform gradient accumulation update
            train_loss = train_loss / self.grad_accum_steps
            train_loss.backward()
            # 梯度裁剪,防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == dataset_size):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            # 定期报告训练损失
            if step % report_loss_per_steps == 0:                
                mean_train_loss = np.mean(train_losses)
                print(f'Step {step} - mean train loss {mean_train_loss:>9f}')

        # 记录训练信息
        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        return logs, train_losses

    def train_step(self, batch):
        """
        执行一个训练步骤。

        Args:
            batch: 训练数据批次

        Returns:
            loss: 训练损失
        """
        # 处理批次数据
        states, actions, returns, timesteps, labels = process_batch(batch, device=self.device)
        # 前向传播
        actions_pred = self.model(states, actions, returns, timesteps)
        actions_pred = actions_pred.permute(0, 2, 1)
        # 计算损失
        loss = self.loss_fn(actions_pred, labels)
        return loss
