import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class accr():
    def __init__(self):

        self.init_q1 = 0
        self.init_q2 = 0

        self.epixilong = 0
        self.greedy = None

    
    def Q1_cal(self):
        # Define the first lever's Gaussian distribution
        mu1, sigma1 = 6, np.sqrt(15)
        # print('21',np.random.normal(mu1, sigma1))
        return np.random.normal(mu1, sigma1)

    def Q2_cal(self):
        # Define the second lever's Mixture of Gaussians
        mu21, sigma21 = 11, np.sqrt(16)
        mu22, sigma22 = 3, np.sqrt(8)
        if np.random.rand() < 0.5:
            # print('21',np.random.normal(mu21, sigma21))
            return np.random.normal(mu21, sigma21)
        else:
            # print('22',np.random.normal(mu22, sigma22))
            return np.random.normal(mu22, sigma22)


    def acc_cacl(self, policy_greeedy, Q1_value ,Q2_value):
        self.epixilong = policy_greeedy
        Q1_sum = 0
        Q2_sum = 0
        for _ in range(100):
            list_ACC = []
            Q1 = Q1_value
            Q2 = Q2_value
            Q1_before = 0
            Q2_before = 0
            self.choose_q1 = None

            ACC =0 

            
            for k in range(1000):
                k += 1
                # policy select
                if random.random() < 1-self.epixilong:
                    self.greedy = True
                else:
                    self.greedy = False

                if self.greedy:
                    # greedy
                    if Q1 >= Q2:
                        q = self.Q1_cal()
                        self.choose_q1 = True
                    else:
                        q = self.Q2_cal()
                else:
                    # Random
                    if random.random() < 0.5:
                        q = self.Q1_cal()
                        self.choose_q1 = True
                    else:
                        q = self.Q2_cal()
                #！！！！！！！！！！！！define alpha！！！！！！！！！！！！！！！！！
                # alpha = 1
                # alpha = 0.9 ** k
                # alpha = 1 / (1 + np.log(1 + k))
                # alpha= 1/k
                alpha = 0.1

                if self.choose_q1:
                    Q1 = Q1 + alpha*(q-Q1_before)
                    Q2 = Q2 
                    Q1_before = Q1
                    r = Q1
                    self.choose_q1 = False
                else:
                    Q1 = Q1 
                    Q2 = Q2 + alpha*(q-Q2_before)
                    Q2_before = Q2
                    r = Q2
                
                ACC += r
                average_ACC = ACC/k
                # print('k',k,'r',r,'avg_ACC',average_ACC)
                list_ACC.append(average_ACC)

            Q1_sum += Q1
            Q2_sum += Q2
            # print('Q1 ', Q1)
            # print('Q1 & Q2', Q1, Q2)
        print('SUM', Q1_sum, Q2_sum)
        return list_ACC, Q1_sum/100, Q2_sum/100
    

if __name__ == '__main__':
    test = accr()
    list1, Q1_average_100, Q2_average_100 = test.acc_cacl(policy_greeedy  = 0.1, Q1_value = 0 ,Q2_value =0)
    print('Epsilon  = 0.0',Q1_average_100, Q2_average_100)
    list2, Q1_average_100, Q2_average_100 = test.acc_cacl(policy_greeedy  = 0.1, Q1_value = 6, Q2_value =7)
    print('Epsilon  = 0.1',Q1_average_100, Q2_average_100)
    list3, Q1_average_100, Q2_average_100 = test.acc_cacl(policy_greeedy  = 0.1, Q1_value = 15, Q2_value =15)
    print('Epsilon  = 0.2',Q1_average_100, Q2_average_100)
   

    plt.figure(figsize=(10, 6))
    
    # 依次绘制4条曲线，每条曲线用一个 label 标识，以便在图例中显示
    plt.plot(list1, marker='o', markersize=3, linewidth=1, label='Q=[0,0]')
    plt.plot(list2, marker='^', markersize=3, linewidth=1, label='Q=[6,7]')
    plt.plot(list3, marker='s', markersize=3, linewidth=1, label='Q=[15,15]')
    

    plt.title('Average Accumulated Reward')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    # 显示图例（根据 label 的值自动生成）
    plt.legend()

    # 显示图形
    plt.show()