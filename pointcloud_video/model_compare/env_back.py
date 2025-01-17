import numpy as np
import Hyperparameters
RANDOM_SEED = Hyperparameters.RANDOM_SEED
#每个gof的时间!!!!!!!!!!!!!!!!!!
VIDEO_GOF_LEN = Hyperparameters.VIDEO_GOF_LEN #秒
#每个GOF有N个'F' 30
F_IN_GOF=Hyperparameters.F_IN_GOF
#一个点云切块2*3*2
TILE_IN_F=Hyperparameters.TILE_IN_F
PACKET_PAYLOAD_PORTION =Hyperparameters.PACKET_PAYLOAD_PORTION
DECODING_TIME_RATIO=Hyperparameters.DECODING_TIME_RATIO
FRAME=Hyperparameters.FRAME
class Environment:
    def __init__(self, cooked_time, cooked_bw,video_size,random_seed=RANDOM_SEED):
        
        np.random.seed(random_seed)

        self.cooked_time = cooked_time
        self.cooked_bw = cooked_bw

        self.video_frame_counter = 0
        self.buffer_size = 0
        #这些指针用于遍历预先定义的网络条件数据（存储在 cooked_time 和 cooked_bw 中）。
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1
        #记录了上一个网络条件更新的时间点。 
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_size=video_size
        self.buffer=[]
        #初始化缓冲区，缓冲区的大小为视频帧数除以每个GOF的帧数，每个GOF的缓冲区大小为tile的数量。
        for i in range(FRAME//F_IN_GOF):
            self.buffer.append([])
            for j in range(TILE_IN_F):
                self.buffer[i].append(-1)        
                
    # 计算下载一个视频GOF所需的时间，并更新缓冲区
    def get_video_gof_new(self, selected_tile,selected_quality):
        delay = 0.0  
        # 初始化下载计数器
        video_gof_counter_sent = 0  
        # 初始化当前GOF的大小
        cur_gof_size=0
        #遍历当前GOF的每个帧，并计算每个帧的缓冲区大小
        for frame in range(F_IN_GOF):
            for tile in range(TILE_IN_F):
                # 如果tile可见，则累加视频大小
                if selected_tile[tile]>0.1:
                    cur_gof_size+=self.video_size[self.video_frame_counter+frame][tile][selected_quality[tile]]
        # print(cur_gof_size,tcnt)    
        # 加上解码时间
        delay+=cur_gof_size*DECODING_TIME_RATIO#decoding time
        # 遍历网络条件数据，模拟下载视频GOF的过程
        while True:  # download video chunk over mahimahi
            # 获取当前网络的吞吐量
            throughput = self.cooked_bw[self.mahimahi_ptr]
            # 计算当前网络吞吐量下的下载时间
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
            # 计算当前网络吞吐量下的下载数据量
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            # 如果当前网络吞吐量下的下载大小超过了当前GOF的大小，则计算剩余时间并退出循环
            if video_gof_counter_sent + packet_payload > cur_gof_size:
                # 
                fractional_time=(cur_gof_size-video_gof_counter_sent)/throughput/PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            # 更新下载计数器和延迟
            video_gof_counter_sent += packet_payload
            delay += duration
            # 更新上一个网络条件更新的时间点
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            # 移动到下一个网络条件数据
            self.mahimahi_ptr += 1

            # 如果网络条件数据遍历完毕，则循环回到开始
            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
                pass
        # 更新缓冲区
        for tile in range(TILE_IN_F):
            if selected_tile[tile]>0.1:
                self.buffer[int(self.video_frame_counter/F_IN_GOF)][tile]=selected_quality[tile]
        self.video_frame_counter += F_IN_GOF
        # 判断是否到达视频末尾
        end_of_video = False
        if self.video_frame_counter>= len(self.video_size)-1:
            end_of_video = True
        return delay,self.buffer
    
    # 计算下载一个视频GOF所需的时间，并更新缓冲区
    def get_video_gof_back(self, selected_gof,selected_tile,selected_quality):
        #现在假设每个gof内的每个frame：quality不变，且不同的tile选用相同的quality

        delay = 0.0  
        video_gof_counter_sent = 0  
        cur_gof_size=0
        for frame in range(F_IN_GOF):
            for tile in range(TILE_IN_F):
                if selected_tile[tile]>0.1:
                    cur_gof_size+=self.video_size[selected_gof*F_IN_GOF+frame][tile][selected_quality[tile]]
        # print(cur_gof_size,tcnt)    
        delay+=cur_gof_size*DECODING_TIME_RATIO#decoding time
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr]
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_gof_counter_sent + packet_payload > cur_gof_size:
                fractional_time=(cur_gof_size-video_gof_counter_sent)/throughput/PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_gof_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
                pass
        for j in range(TILE_IN_F):
            if selected_tile[j]==1:
                self.buffer[selected_gof][j]=selected_quality[j]
        return delay,self.buffer
    def predict_bw(self):
        return self.cooked_bw[self.mahimahi_ptr-1]*0.85+self.cooked_bw[self.mahimahi_ptr]*0.15