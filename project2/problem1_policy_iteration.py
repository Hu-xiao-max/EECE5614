import numpy as np
import random
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class policy:
    def __init__(self):
        
        self.p = 0.5
        self.gamma= 0.99
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

        self.index_matrix = np.array([[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0],
                         [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 0],
                         [0,  19,  20,  21,  22,   0,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35, 0],
                         [0,  36,  37,  38,  39,   0,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52, 0],
                         [0,  53,  54,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  55,  56, 0],
                         [0,  57,  58,   0,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73, 0],
                         [0,  74,  75,   0,  76,  77,   0,  78,  79,   0,  80,  81,  82,  83,  84,   0,  85,  86,  87, 0],
                         [0,  88,  89,   0,  90,  91,   0,  92,  93,   0,  94,  95,   0,   0,   0,   0,  96,  97,  98, 0],
                         [0,  99, 100, 101, 102, 103,   0, 104, 105,   0, 106, 107, 108, 109, 110,   0, 111, 112, 113, 0],
                         [0, 114, 115, 116, 117, 118,   0, 119, 120,   0, 121, 122, 123, 124, 125,   0, 126, 127, 128, 0],
                         [0,   0,   0,   0,   0, 129,   0, 130, 131,   0,   0, 132, 133, 134, 135,   0, 136, 137, 138, 0],
                         [0, 139, 140, 141, 142, 143,   0, 144, 145, 146,   0, 147, 148,   0, 149,   0,   0,   0, 150, 0],
                         [0, 151, 152,   0,   0,   0,   0,   0, 153, 154,   0, 155, 156,   0, 157, 158, 159,   0, 160, 0],
                         [0, 161, 162, 163, 164, 165, 166,   0, 167, 168,   0, 169, 170,   0, 171, 172, 173,   0, 174, 0],
                         [0, 175, 176, 177, 178, 179, 180,   0, 181, 182,   0, 183, 184,   0, 185, 186, 187, 188, 189, 0],
                         [0, 190, 191, 192, 193, 194, 195,   0, 196, 197, 198, 199, 200,   0,   0,   0,   0, 201, 202, 0],
                         [0, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 0],
                         [0,   0,   0, 221, 222, 223, 224,   0,   0,   0,   0,   0,   0, 225, 226, 227, 228, 229, 230, 0],
                         [0, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 0],
                         [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0]])

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
        # initial policy all left
        before_value = np.full([20, 20], 0)
        before_Action = np.full([20, 20], 13)# init policy all left
        count_step = 0
        while True:
            count_step +=1
            while True:
                for ev_i in range(self.env.shape[0]):  # Iterate over rows
                    for ev_j in range(self.env.shape[1]):  # Iterate over columns
                        if not self.env[ev_i,ev_j] == 1 :
                            # sum p(s' | s a)[R(s,a,s')+gamma V_{n-1}(s)]
                            if self.init:
                                pick_action = 13
                                self.Value[ev_i, ev_j] = self.value_action(pick_action, ev_i, ev_j, before_value)
                                
                            else:
                                pick_action = self.Action[ev_i, ev_j]
                                self.Value[ev_i, ev_j] = self.value_action(pick_action, ev_i, ev_j, before_value)
                            
                            self.Value[3,13] = 0# 终点为0！importatn!!

                if np.max(np.abs(self.Value-before_value)) < self.theta:
                    break
                before_value = copy.deepcopy(self.Value)
            self.init = False 

            # Policy Improvement
            for improve_i in range(self.env.shape[0]):  # Iterate over rows
                    for improve_j in range(self.env.shape[1]):  # Iterate over columns
                        if not self.env[improve_i, improve_j]  == 1 :
                            # sum p(s' | s a)[R(s,a,s')+gamma V_{n-1}(s)]
                            v_action_list = []
                            for action_slect in self.action_list:# up,down,left,right
                                v_for_max_action = self.value_action(action_slect, improve_i, improve_j, before_value)
                                v_action_list.append(v_for_max_action)
                            max_action = self.action_list[v_action_list.index(max(v_action_list))]
                            self.Action[improve_i, improve_j] = max_action
                    
            if np.all(self.Action == before_Action):
                break
            before_Action = copy.deepcopy(self.Action)


         
            
        State_Matrix = self.Value.astype(int)
        # State_Matrix = self.Value
        print(State_Matrix)

        annot_matrix = np.where(self.env == 1, "", State_Matrix)

        plt.figure(figsize=(15, 10))
        sns.heatmap(self.env,fmt="",  cmap=sns.color_palette([self.colors[i] for i in range(6)]), cbar=False,annot=annot_matrix, linewidths=0.5, linecolor='black')
        plt.axis('off')
        plt.title('Maze Problem - State Numbers')
        plt.show()


        # directions = ["up", "right", "down", "left"]
        # [11,12,13,14] up,down,left,right
        # directions = [11, 14, 12, 13]
        policy = self.Action.flatten()
        print('!!!!!!!!!!!!!!!!!', policy)
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
        print('count_step', count_step)
                

                    
                                            

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
            if self.env[o_i, o_j] == 2:
                reward += -5
            elif self.env[o_i, o_j] == 3:
                reward += -10
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



        
        
        










