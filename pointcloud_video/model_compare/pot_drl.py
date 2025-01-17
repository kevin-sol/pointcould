import numpy as np
import env_back as env
import load_data
import Hyperparameters
import math
import fov_predict
import os
from Config import Config
import scddqn
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
VIDEO_GOF_LEN=Hyperparameters.VIDEO_GOF_LEN
QUALITY_LEVELS =  Hyperparameters.QUALITY_LEVELS
REBUF_PENALTY = Hyperparameters.REBUF_PENALTY  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = Hyperparameters.SMOOTH_PENALTY
DEFAULT_QUALITY = Hyperparameters.DEFAULT_QUALITY  # default video quality without agent
RANDOM_SEED = Hyperparameters.RANDOM_SEED
RESEVOIR = 1  # BB
CUSHION = 3  # BB
LOG_FILE = current_directory+'/results/log_sim_pot_drl'
RESULT_FILE=current_directory+'/results/result_sim_pot_drl'
INIT_QOE=Hyperparameters.INIT_QOE
MULTIPLE_QUALITY_LEVELS=Hyperparameters.MULTIPLE_QUALITY_LEVELS
F_IN_GOF=Hyperparameters.F_IN_GOF
TILE_IN_F=Hyperparameters.TILE_IN_F
FRAME=Hyperparameters.FRAME
config = Config()
N = config.N

tile_num = config.tile_num  # 切块数量
level_num = config.level_num  # 质量等级数量
group_of_x = config.group_of_x  # 决策变量x的组数（5个一组）-
c_num = config.c_num
x_num = config.x_num
s_len = config.s_len
memory_size = config.memory_size
batch_size = config.batch_size
#######DQN参数#######
num_episodes = config.num_episodes             # 训练的总episode数量
num_exploration_episodes = config.num_exploration_episodes  # 探索过程所占的episode数量
# max_len_episode = 1500          # 每个episode的最大回合数
batch_size = config.batch_size                # 批次大小
learning_rate = config.learning_rate            # 学习率
gamma = config.gamma                      # 折扣因子
initial_epsilon = config.initial_epsilon            # 探索起始时的探索率
final_epsilon = config.final_epsilon           # 探索终止时的探索率
def test(cooked_time,cooked_bw,video_size,fov,dis,time=None):

    np.random.seed(RANDOM_SEED)

    net_env = env.Environment(cooked_time=cooked_time,
                              cooked_bw=cooked_bw,
                              video_size=video_size
                              )

    log_path = LOG_FILE+str(time)
    log_file = open(log_path, 'w')
    log_file.write('time_stamp' + '\t' +'play_fov' + '\t' +
                       'option' + '\t'+
                       'sum_rebuffer' +  '\t'+
                       'selected_tile' +  '\t'+
                       'bit_rate' + '\n')
    time_stamp = 0

    last_bit_rate = [DEFAULT_QUALITY]*TILE_IN_F
    bit_rate = [DEFAULT_QUALITY]*TILE_IN_F
    player_fov_count=0
    fov_count = 0
    player_fractional_time=0
    sum_reward_quality=0
    sum_reward_rebuffer=0
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
        if fov_count-player_fov_count-player_fractional_time/VIDEO_GOF_LEN>0 and player_fov_count+player_fractional_time>0 and 1==0: #back
            for predict_window in range(player_fov_count+math.ceil(player_fractional_time/VIDEO_GOF_LEN),fov_count):
                predicted_tile=fov_predict.predict(time,int(player_fov_count*F_IN_GOF+player_fractional_time*F_IN_GOF/VIDEO_GOF_LEN),predict_window*F_IN_GOF,(predict_window+1)*F_IN_GOF-1)
                predicted_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*predicted_tile)]
                selected_tile=[0]*TILE_IN_F
                for tile in range(TILE_IN_F):
                    if predicted_tile[tile]==1 and buffer[predict_window][tile]==-1:
                        selected_tile[tile]=1
                if sum(selected_tile)>0:
                    back=1
                    delay,buffer=net_env.get_video_gof_back(predict_window,selected_tile,[0]*TILE_IN_F)
                    time_stamp+=delay
                    if player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay<predict_window*VIDEO_GOF_LEN: #不rebuffer
                        player_fov_count+=int((player_fractional_time+delay)//VIDEO_GOF_LEN)
                        player_fractional_time=(player_fractional_time+delay)-int((player_fractional_time+delay)//VIDEO_GOF_LEN)*VIDEO_GOF_LEN
                    else:
                        sum_reward_rebuffer+=(player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay-predict_window*VIDEO_GOF_LEN)*REBUF_PENALTY
                        player_fov_count=predict_window
                        player_fractional_time=0
                    log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                            'back' +str(predict_window)+ '\t'+
                            str(round(sum_reward_rebuffer,1)) + '\t'+ 
                            str(selected_tile) + '\t'+ 
                            str(bit_rate)+'\n')    
                    break
            if back==0 and fov_count==FRAME//F_IN_GOF and fov_count>player_fov_count:#
                if player_fractional_time+0.25>=VIDEO_GOF_LEN:
                    player_fractional_time=0
                    player_fov_count+=1
                    time_stamp+=VIDEO_GOF_LEN-player_fractional_time
                else:
                    time_stamp+=0.25
                    player_fractional_time+=0.25
                continue
            if back:
                continue
        if back==0 and fov_count<FRAME//F_IN_GOF:#new 
            #如果距离上次播放超过2.3s，则播放下一个gof
            if fov_count-player_fov_count-player_fractional_time/VIDEO_GOF_LEN>2.3:
                if player_fractional_time+0.1>=VIDEO_GOF_LEN:
                    player_fov_count+=1
                    time_stamp+=VIDEO_GOF_LEN-player_fractional_time
                    player_fractional_time=0
                else:
                    player_fractional_time+=0.1
                    time_stamp+=0.1
                continue
            selected_tile=fov_predict.predict(time,max(int(player_fov_count*F_IN_GOF+player_fractional_time*F_IN_GOF/VIDEO_GOF_LEN),(fov_count-2)*F_IN_GOF),fov_count*F_IN_GOF,(fov_count+1)*F_IN_GOF-1)
            # selected_tile=fov[fov_count*F_IN_GOF:(fov_count+1)*F_IN_GOF]
            selected_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*selected_tile)]
            state=[]        
            for i in range(TILE_IN_F):
                for j in range(N):
                    for k in range(level_num):
                        if fov_count*F_IN_GOF+j>=len(video_size):
                            state.append(video_size[-1][i][k])
                        else:
                            state.append(video_size[fov_count*F_IN_GOF+j][i][k])
            # print(len(state))
            for j in range(N):
                state+=selected_tile
            # print(len(state))
            for s in range(x_num):
                state.append(0)
            # print(len(state))
            state+=selected_tile
            # print(len(state))
            if fov_count+N>=len(cooked_bw):
                state+=cooked_bw[-1]*N
            else:
                state+=(cooked_bw[fov_count:fov_count+N])
            # print(len(state))
            for i in last_bit_rate:
                lbr=[0]*level_num
                lbr[j]=1
                state+=(lbr)
            # print(len(state))
            buffer_size=fov_count*VIDEO_GOF_LEN-(player_fov_count*VIDEO_GOF_LEN+player_fractional_time)
            state.append(buffer_size)
            # print(len(state))
            for i in range(TILE_IN_F):
                for j in range(N):
                    if fov_count*F_IN_GOF+j>=len(dis):
                        state.append(dis[-1][i])
                    else:
                        state.append(dis[fov_count*F_IN_GOF+j][i])
            # print(len(state))

            bit_rate=scddqn.get_bitrate(state)

            delay,buffer=net_env.get_video_gof_new(selected_tile,bit_rate)
            if player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay<fov_count*VIDEO_GOF_LEN:
                player_fov_count+=int((player_fractional_time+delay)//VIDEO_GOF_LEN)
                player_fractional_time=(player_fractional_time+delay)-int((player_fractional_time+delay)//VIDEO_GOF_LEN)*VIDEO_GOF_LEN
            else:
                sum_reward_rebuffer+=(player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay-fov_count*VIDEO_GOF_LEN)*REBUF_PENALTY
                player_fov_count=fov_count
                player_fractional_time=0
            time_stamp+=delay
            log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                            'new' +str(fov_count)+ '\t'+
                            str(round(sum_reward_rebuffer,1)) + '\t'+ 
                            str(selected_tile) + '\t'+ 
                            str(bit_rate)+'\n')  
            fov_count+=1
            continue
        break       
    tp=tn=fp=fn=0 
    seen_tile=0
    last_bit_rate=[0]*TILE_IN_F
    for fov_count in range(int(FRAME/F_IN_GOF)):
        seen=[0]*TILE_IN_F
        for s in range(TILE_IN_F):
            sum_reward_switch+=SMOOTH_PENALTY * np.abs(MULTIPLE_QUALITY_LEVELS[max(buffer[fov_count][s],0)]-MULTIPLE_QUALITY_LEVELS[last_bit_rate[s]])
            last_bit_rate[s]=max(buffer[fov_count][s],0)
            for ff in range(F_IN_GOF):
                if fov[fov_count*F_IN_GOF+ff][s]:
                    seen[s]=1
                    continue
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
    return [sum_reward_quality/len(video_size),sum_reward_rebuffer/len(video_size),sum_reward_switch/len(video_size)]