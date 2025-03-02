import numpy as np
import random
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class policy:
    def __init__(self):
        
        self.p = 0.025
        self.gamma= 0.5
        self.theta = 0.01
        self.env = self.env_define()
        self.Value = np.zeros([20, 20])
        self.Action = np.zeros([20, 20])
        self.action_list= [11,12,13,14]#up,down,left,right


        self.init = True


        # Define colors
        self.colors = {
            0: [1, 1, 1],        # White
            1: [0, 0, 0],        # Black
            2: [0.55, 0, 0],     # Light Brown
            3: [0.96, 0.8, 0.6], # Dark Red
            4: [0, 0, 1],        # Green
            5: [0, 1, 0]         # Blue
        }

        pass

    def env_define(self):
        Env = np.zeros([20, 20])   # The matrix which describe the environment
        # regular = 0    wall = 1    oil = 2     bump = 3    start = 4    end = 5
        Env[:, [0]] = np.ones([20, 1])
        Env[:, [19]] = np.ones([20, 1])
        Env[0, :] = np.ones([1, 20])
        Env[19, :] = np.ones([1, 20])
        Env[1, 11] = 3; Env[1, 12] = 3
        Env[2, 1:4] = [3, 3, 3]; Env[2, 5] = 1; Env[2, 8] = 2; Env[2, 16] = 2
        Env[3, 5] = 1; Env[3, 13] = 5
        Env[4, 2] = 2; Env[4, 3:17] = np.ones([1, 14])
        Env[5, 1] = 3; Env[5, 3] = 1; Env[5, 6] = 2; Env[5, 9] = 3; Env[5, 17] = 3
        Env[6, 3] = 1; Env[6, 6] = 1; Env[6, 9] = 1; Env[6, 15] = 1; Env[6, 17] = 3
        Env[7, 2] = 3; Env[7, 3] = 1; Env[7, 6] = 1; Env[7, 9] = 1; Env[7, 10:12] = [3, 3]; Env[7, 12:16] = np.ones([1, 4]); Env[7, 17] = 3
        Env[8, 6] = 1; Env[8, 9] = 1; Env[8, 15] = 1; Env[8, 17] = 3
        Env[9, 6] = 1; Env[9, 9] = 1; Env[9, 15] = 1
        Env[10, 1:5] = np.ones([1, 4]); Env[10, 6] = 1; Env[10, 9:11] = [1, 1]; Env[10, 15] = 1; Env[10, 18] = 2
        Env[11, 6] = 1; Env[11, 10] = 1; Env[11, 13] = 1; Env[11, 15:18] = [1, 1, 1]
        Env[12, 3:8] = np.ones([1, 5]); Env[12, 10] = 1; Env[12, 11:13] = [3, 3]; Env[12, 13] = 1; Env[12, 17] = 1
        Env[13, 7] = 1; Env[13, 10] = 1; Env[13, 13] = 1; Env[13, 17] = 1
        Env[14, 1:3] = [3, 3]; Env[14, 7] = 1; Env[14, 10] = 1; Env[14, 13] = 1
        Env[15, 4] = 4; Env[15, 7] = 1; Env[15, 10] = 2; Env[15, 13:17] = [1, 1, 1, 1]; Env[15, 17:19] = [3, 3]
        Env[16, 7] = 3; Env[16, 10] = 2
        Env[17, 1:3] = [1, 1]; Env[17, 7:13] = [1, 1, 1, 1, 1, 1]; Env[17, 14] = 2; Env[17, 17] = 2
        Env[18, 7] = 2

        return Env
    

    def run_policy(self):

        '''
        start: [15, 4]
        End:[3, 13]
        '''
        
        # 1-p move to anticipated state 2/p to prependicular

        before_value = np.full([20, 20], 0)


        while True:
            for i in range(self.env.shape[0]):  # Iterate over rows
                for j in range(self.env.shape[1]):  # Iterate over columns
                    if not self.env[i, j] == 1 :
                        # sum p(s' | s a)[R(s,a,s')+gamma V_{n-1}(s)]
                        v_action_list = []
                        for action_slect in self.action_list:# up,down,left,right
                            v_for_max_action = self.value_action(action_slect, i, j, before_value)
                            v_action_list.append(v_for_max_action)
                        max_action = self.action_list[v_action_list.index(max(v_action_list))]
                        self.Action[i, j] = max_action

                        self.Value[i, j] = max(v_action_list)
            self.Value[3,13] = 0# 终点为0
            

            if np.max(np.abs(self.Value-before_value)) < self.theta:
                break
            before_value = copy.deepcopy(self.Value)
            
         
            
        State_Matrix = self.Value.astype(int)
        print(State_Matrix)

        annot_matrix = np.where(self.env == 1, "", State_Matrix)

        plt.figure(figsize=(15, 10))
        sns.heatmap(self.env,fmt="",  cmap=sns.color_palette([self.colors[i] for i in range(6)]), cbar=False,annot=annot_matrix, linewidths=0.5, linecolor='black')
        plt.axis('off')
        plt.title('Maze Problem - State Numbers')
        plt.show()


        policy = self.Action.flatten()
        # print('!!!!!!!!!!!!!!!!!', policy)
        # print(len(policy))

        # 将数据重塑为 20x20 矩阵
        data = policy.reshape(20, 20)

        # 创建绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 19.5)
        ax.set_ylim(-0.5, 19.5)
        ax.set_frame_on(False)

        # 绘制背景颜色
        for i in range(20):
            for j in range(20):
                color = self.colors[int(self.env[i, j])]
                ax.add_patch(plt.Rectangle((j - 0.5, 19 - i - 0.5), 1, 1, color=color, ec='gray'))

        # 定义箭头方向
        arrow_map = {11: (0, 0.3), 12: (0, -0.3), 13: (-0.3, 0), 14: (0.3, 0)}

        # 绘制箭头
        for i in range(20):
            for j in range(20):
                value = data[i, j]
                if value in arrow_map:
                    dx, dy = arrow_map[value]
                    ax.arrow(j, 19 - i, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')

        # 显示图像
        plt.show()

                
                    
                                            

    def value_action(self,pick_action, i, j, before_value):
        # policy Evalution
        p = self.p
        next_i1, next_j1 = self.action_take(pick_action, i, j, 0)
        real_i1, real_j1, reward1 = self.next_state_reward(i, j, next_i1, next_j1)
        value1 = (1-p)*(reward1+self.gamma*before_value[real_i1, real_j1])

        next_i2, next_j2 = self.action_take(pick_action, i, j, 1)
        real_i2, real_j2, reward2 = self.next_state_reward(i, j, next_i2, next_j2)
        value2 = (p/2)*(reward2+self.gamma*before_value[real_i2, real_j2])

        next_i3, next_j3 = self.action_take(pick_action, i, j, 2)
        real_i3, real_j3, reward3 = self.next_state_reward(i, j, next_i3, next_j3)
        value3 = (p/2)*(reward3+self.gamma*before_value[real_i3, real_j3])

        value = value1+value2+value3

        return value

                            

    def transfer(self):

        p = self.p
        prob_do = 1 - p # 0
        prob_left = p / 2# 1
        prob_right = p / 2# 2

        list_p = [prob_do, prob_left, prob_right]
        result = random.choices([0, 1, 2], weights=[prob_do, prob_left, prob_right])[0]
        # return acutaly state with probability
        # list_p[result]
        return result
            
    def action_take(self,action, t_i, t_j, result):
        o_i = t_i
        o_j = t_j
        if action == 11:
            #up
            if result == 0:
                #do 
                t_i = t_i -1
                t_j = t_j
            elif result == 1:
                # left
                t_i = t_i 
                t_j = t_j -1

            elif result == 2:
                # right
                t_i = t_i 
                t_j = t_j +1

        if action == 12:
            #down
            if result == 0:
                #do 
                t_i = t_i +1
                t_j = t_j
            elif result == 1:
                # left
                t_i = t_i 
                t_j = t_j -1

            elif result == 2:
                # right
                t_i = t_i 
                t_j = t_j +1

        if action == 13:
            #left
            if result == 0:
                # print('do')
                #do 
                t_i = t_i 
                t_j = t_j -1
            elif result == 1:
                # print('up')
                # up
                t_i = t_i -1
                t_j = t_j 

            elif result == 2:
                # print('down')
                # down
                t_i = t_i +1
                t_j = t_j 

        if action == 14:
            #right
            if result == 0:
                #do 
                t_i = t_i 
                t_j = t_j +1
            elif result == 1:
                # left
                t_i = t_i -1
                t_j = t_j 

            elif result == 2:
                # right
                t_i = t_i +1
                t_j = t_j 

        t_i, t_j = self.index_check(o_i, o_j, t_i, t_j)

        return t_i, t_j
    
    def index_check(self, o_i, o_j, row, col):
        if not 0 <= row < self.env.shape[0]:
            return o_i, o_j
        if not 0 <= col < self.env.shape[1]:
            return o_i, o_j
        return row, col
        


    def next_state_reward(self, o_i, o_j, t_i, t_j):
        # regular = 0    wall = 1    oil = 2     bump = 3    start = 4    end = 5
        if self.env[t_i, t_j] == 1:
            # print('wall')
            # 撞墙就返回
            reward = -1.8
            return o_i, o_j, reward
        
        elif self.env[t_i, t_j] == 2:
            # print('oil')
            reward = -6
            return t_i, t_j, reward
        
        elif self.env[t_i, t_j] == 3:
            # print('bump')
            reward = -11
            return t_i, t_j, reward
        
        elif self.env[t_i, t_j] == 5:
            reward = 299
            # print('get end!!!')
            return t_i, t_j, reward
        
        else: 
            reward = -1
            return t_i, t_j, reward
        


if __name__ == '__main__':
    instance_policy = policy()
    instance_policy.run_policy()