import random
import matplotlib.pyplot as plt


class accr():
    def __init__(self):
        self.init_q = 0
        self.q1 = 6
        self.q2 = 7
        self.epixilong = 0
        self.greedy = None

    def acc_cacl(self, alpha, policy_greeedy):
        self.epixilong = policy_greeedy
        list_ACC = []
        Q1 = self.init_q
        Q2 = self.init_q
        Q1_before = 0
        Q2_before = 0
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
                if Q1 > Q2:
                    q = self.q1
                else:
                    q = self.q2
            else:
                # Random
                if random.random() < 0.5:
                    q = self.q1
                else:
                    q = self.q2
            
            if q == self.q1:
                Q1 = Q1 + alpha*(q-Q1_before)
                Q2 = Q2 
                Q1_before = Q1
                r = Q1
            else:
                Q1 = Q1 
                Q2 = Q2 + alpha*(q-Q2_before)
                Q2_before = Q2
                r = Q2
            
            ACC += r
            average_ACC = ACC/k
            print('k',k,'r',r,'avg_ACC',average_ACC)
            list_ACC.append(average_ACC)
        return list_ACC
    

if __name__ == '__main__':
    test = accr()

    list1 = test.acc_cacl(alpha=1, policy_greeedy  = 0.0)
    list2 = test.acc_cacl(alpha=1, policy_greeedy  = 0.1)
    list3 = test.acc_cacl(alpha=1, policy_greeedy  = 0.2)
    list4 = test.acc_cacl(alpha=1, policy_greeedy  = 0.5)

    plt.figure(figsize=(10, 6))
    
    # 依次绘制4条曲线，每条曲线用一个 label 标识，以便在图例中显示
    plt.plot(list1, marker='o', markersize=3, linewidth=1, label='alpha=0.0')
    plt.plot(list2, marker='^', markersize=3, linewidth=1, label='alpha=0.1')
    plt.plot(list3, marker='s', markersize=3, linewidth=1, label='alpha=0.2')
    plt.plot(list4, marker='x', markersize=3, linewidth=1, label='alpha=0.5')

    plt.title('Comparison of Different Alpha Settings')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    # 显示图例（根据 label 的值自动生成）
    plt.legend()

    # 显示图形
    plt.show()