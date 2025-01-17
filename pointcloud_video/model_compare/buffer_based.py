import numpy as np
import env_back as env
import load_data
import Hyperparameters
import math
import fov_predict
import os
from Config import Config
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
VIDEO_GOF_LEN=Hyperparameters.VIDEO_GOF_LEN
QUALITY_LEVELS =  Hyperparameters.QUALITY_LEVELS
REBUF_PENALTY = Hyperparameters.REBUF_PENALTY  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = Hyperparameters.SMOOTH_PENALTY
DEFAULT_QUALITY = Hyperparameters.DEFAULT_QUALITY  # default video quality without agent
RANDOM_SEED = Hyperparameters.RANDOM_SEED
RESEVOIR = 1  # BB
CUSHION = 3  # BB
LOG_FILE = current_directory+'/results/log_sim_bb'
RESULT_FILE=current_directory+'/results/result_sim_bb'
INIT_QOE=Hyperparameters.INIT_QOE
MULTIPLE_QUALITY_LEVELS=Hyperparameters.MULTIPLE_QUALITY_LEVELS
F_IN_GOF=Hyperparameters.F_IN_GOF
TILE_IN_F=Hyperparameters.TILE_IN_F
FRAME=Hyperparameters.FRAME
def test(cooked_time,cooked_bw,video_size,fov,dis,time=None):

    np.random.seed(RANDOM_SEED)

    net_env = env.Environment(cooked_time=cooked_time,
                              cooked_bw=cooked_bw,
                              video_size=video_size
                              )

    log_path = LOG_FILE+str(time)
    log_file = open(log_path, 'w')
    log_file.write('time_stamp' + '\t' +'play_fov' + '\t'  +
                       'option' + '\t'+
                       'sum_rebuffer' +  '\t'+
                       'selected_tile' +  '\t'+
                       'bit_rate' + '\n')
    time_stamp = 0

    last_bit_rate = [DEFAULT_QUALITY]*TILE_IN_F
    bit_rate = [DEFAULT_QUALITY]*TILE_IN_F
    #这个变量用于记录播放器已经播放的完整帧组（GOF）的数量。
    # 它基本上表示已经处理并播放的完整GOF的计数 
    player_fov_count=0
    #预测窗口
    #这个变量用于跟踪已经预测或处理过的GOF数量，代表了基于预测的视野（FOV）考虑用于未来播放的GOF数量。
    fov_count = 0
    #这个变量表示当前GOF内的播放时间的小数部分。
    # 它用于跟踪尚未完全播放的GOF内的进度，有助于精确管理播放时间和缓冲计算。
    player_fractional_time=0
    #质量奖励
    sum_reward_quality=0
    #重缓冲惩罚
    sum_reward_rebuffer=0
    #切换奖励
    sum_reward_switch=0
    #首个gof，全选第一个f的
    selected_tile=[fov[0]]*F_IN_GOF
    selected_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*selected_tile)]
    buffer=[]
    for i in range(FRAME//F_IN_GOF):
        buffer.append([])
        for j in range(TILE_IN_F):
            buffer[i].append(-1)
    delay,buffer=net_env.get_video_gof_new(selected_tile,bit_rate)
    fov_count+=1
    time_stamp += delay
    sum_reward_rebuffer+=delay*REBUF_PENALTY
    log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                    'new0' + '\t'+
                    str(round(sum_reward_rebuffer,1)) + '\t'+ 
                    str(selected_tile) + '\t'+ 
                    str(bit_rate)+'\n')
    while True:      
        log_file.flush()
        back=0
        #没有back    
        # 检查是否需要回退下载
        if fov_count-player_fov_count-player_fractional_time/VIDEO_GOF_LEN>0 and player_fov_count+player_fractional_time>0 and 1==0: #back
            # 遍历预测窗口
            for predict_window in range(player_fov_count+math.ceil(player_fractional_time/VIDEO_GOF_LEN),fov_count):
                # 预测tile
                predicted_tile=fov_predict.predict(time,int(player_fov_count*F_IN_GOF+player_fractional_time*F_IN_GOF/VIDEO_GOF_LEN),predict_window*F_IN_GOF,(predict_window+1)*F_IN_GOF-1)
                predicted_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*predicted_tile)]
                selected_tile=[0]*TILE_IN_F
                # 选择需要下载的tile
                for tile in range(TILE_IN_F):
                    if predicted_tile[tile]==1 and buffer[predict_window][tile]==-1:
                        selected_tile[tile]=1
                # 如果有需要下载的tile
                if sum(selected_tile)>0:
                    back=1
                    # 回退下载
                    delay,buffer=net_env.get_video_gof_back(predict_window,selected_tile,[0]*TILE_IN_F)
                    time_stamp+=delay
                    # 更新播放时间
                    if player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay<predict_window*VIDEO_GOF_LEN: #不rebuffer
                        player_fov_count+=int((player_fractional_time+delay)//VIDEO_GOF_LEN)
                        player_fractional_time=(player_fractional_time+delay)-int((player_fractional_time+delay)//VIDEO_GOF_LEN)*VIDEO_GOF_LEN
                    else:
                        # 计算rebuffer时间
                        sum_reward_rebuffer+=(player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay-predict_window*VIDEO_GOF_LEN)*REBUF_PENALTY
                        player_fov_count=predict_window
                        player_fractional_time=0
                    # 记录日志
                    log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                            'back' +str(predict_window)+ '\t'+
                            str(round(sum_reward_rebuffer,1)) + '\t'+ 
                            str(selected_tile) + '\t'+ 
                            str(bit_rate)+'\n')    
                    break
            # 如果到达视频末尾且没有回退下载
            if back==0 and fov_count==FRAME//F_IN_GOF and fov_count>player_fov_count:#
                # 更新播放时间
                if player_fractional_time+0.25>=VIDEO_GOF_LEN:
                    player_fractional_time=0
                    player_fov_count+=1
                    time_stamp+=VIDEO_GOF_LEN-player_fractional_time
                else:
                    time_stamp+=0.25
                    player_fractional_time+=0.25
                continue
            # 如果有回退下载，继续循环
            if back:
                continue
        #如果没有back，且没有到达视频末尾
        if back==0 and fov_count<FRAME//F_IN_GOF:#new 
            # 计算当前缓冲区大小
            buffer_size=fov_count*VIDEO_GOF_LEN-(player_fov_count*VIDEO_GOF_LEN+player_fractional_time)
            
            # 根据缓冲区大小选择比特率
            if buffer_size < RESEVOIR:
                nbit_rate = 0
            elif buffer_size >= CUSHION:
                nbit_rate = QUALITY_LEVELS - 1
            else:
                nbit_rate = (QUALITY_LEVELS - 1) * (buffer_size - RESEVOIR) / float(CUSHION-RESEVOIR)
            nbit_rate = round(nbit_rate)
            bit_rate=[nbit_rate]*TILE_IN_F
            
            # 预测下一个GOF的FOV
            selected_tile=fov_predict.predict(time,int(player_fov_count*F_IN_GOF+player_fractional_time*F_IN_GOF/VIDEO_GOF_LEN),fov_count*F_IN_GOF,(fov_count+1)*F_IN_GOF-1)
            #最终结果是一个一维列表，
            # 其中每个元素代表原来 selected_tile 中每个 tile 的平均值的向上取整。
            #也就是这些帧里只要有1个tile被预测为1，就令selected_tile为1
            selected_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*selected_tile)]
            
            # 获取新的视频GOF
            delay,buffer=net_env.get_video_gof_new(selected_tile,bit_rate)
            
            # 更新播放时间和缓冲状态
            if player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay<fov_count*VIDEO_GOF_LEN:
                player_fov_count+=int((player_fractional_time+delay)//VIDEO_GOF_LEN)
                player_fractional_time=(player_fractional_time+delay)-int((player_fractional_time+delay)//VIDEO_GOF_LEN)*VIDEO_GOF_LEN
            else:
                # 计算rebuffer惩罚
                sum_reward_rebuffer+=(player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay-fov_count*VIDEO_GOF_LEN)*REBUF_PENALTY
                player_fov_count=fov_count
                player_fractional_time=0
            
            # 更新时间戳
            time_stamp+=delay
            
            # 记录日志
            log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                            'new' +str(fov_count)+ '\t'+
                            str(round(sum_reward_rebuffer,1)) + '\t'+ 
                            str(selected_tile) + '\t'+ 
                            str(bit_rate)+'\n')  
            
            # 增加FOV计数
            fov_count+=1
            continue
        break       
    tp=tn=fp=fn=0 
    seen_tile=0
    last_bit_rate=[0]*TILE_IN_F
    best_buf=[]

    for fov_count in range(int(FRAME/F_IN_GOF)):
        seen=[0]*TILE_IN_F
        best_buf.append([])
        for s in range(TILE_IN_F):
            sum_reward_switch+=SMOOTH_PENALTY * np.abs(MULTIPLE_QUALITY_LEVELS[max(buffer[fov_count][s],0)]-MULTIPLE_QUALITY_LEVELS[last_bit_rate[s]])
            last_bit_rate[s]=max(buffer[fov_count][s],0)
            for ff in range(F_IN_GOF):
                if fov[fov_count*F_IN_GOF+ff][s]:
                    seen[s]=1
                    break
            best_buf[fov_count].append(seen[s])
        for i in range(F_IN_GOF):
            for j in range(TILE_IN_F):
                seen_tile+=(buffer[fov_count][j]>=0)*seen[j]*video_size[fov_count*F_IN_GOF+i][j][buffer[fov_count][j]]/dis[fov_count*F_IN_GOF+i][j]
                if buffer[fov_count][j]==-1 and fov[fov_count*F_IN_GOF+i][j]==0:
                    tn+=1
                elif buffer[fov_count][j]>=0 and fov[fov_count*F_IN_GOF+i][j]==1:
                    tp+=1
                elif buffer[fov_count][j]==-1 and fov[fov_count*F_IN_GOF+i][j]==1:
                    fn+=1
                elif buffer[fov_count][j]>=0 and fov[fov_count*F_IN_GOF+i][j]==0:
                    fp+=1

    sum_reward_quality=INIT_QOE*seen_tile

    # log_file.write(str(round(time_stamp,8)) + '\t'+ '\t' + 
    #                 str(round(reward_quality,1)) + '\t'+ '\t'+
    #                 str(round(reward_rebuffer,1)) + '\t'+ '\t'+
    #                 str(round(reward_switch,1)) + '\t'+ '\t'+
    #                 str(round(reward,1)) +  '\t'+ '\t'+
    #                 str(round(buffer_size,1))+  '\t'+ '\t'+
    #                 str(bit_rate)+'\n')
    # log_file.flush()

    result_path = RESULT_FILE
    result_file = open(result_path, 'a')
    result_file.write(str((sum_reward_quality-sum_reward_rebuffer-sum_reward_switch)/len(video_size))+'\n')    
    open(result_path+'quali', 'a').write(str(sum_reward_quality/len(video_size))+'\n')
    open(result_path+'rebuf', 'a').write(str(sum_reward_rebuffer/len(video_size))+'\n') 
    open(result_path+'switch', 'a').write(str(sum_reward_switch/len(video_size))+'\n')
    open(result_path+'acc', 'a').write(str((tp+tn)/(tp+tn+fp+fn))+'\n')
    open(result_path+'recall', 'a').write(str((tp)/(tp+fn))+'\n') 
    open(result_path+'prec', 'a').write(str((tp)/(tp+fp))+'\n') 
    open(result_path+'buffer', 'a').write(str(buffer)+'\n') 
    open(current_directory+'/results/log_buffer', 'a').write(str(best_buf)+'\n') 
    return [sum_reward_quality/len(video_size),sum_reward_rebuffer/len(video_size),sum_reward_switch/len(video_size)]
