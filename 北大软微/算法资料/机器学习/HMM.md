# 隐马尔可夫模型

马尔可夫链：随机概率自动机。 

example：http://www.52nlp.cn/hmm-concrete-example-on-wiki

组成元素：

1. 观测序列：$O_{0...n}$

2. 隐状态：$State_{0....m}$

3. 初始状态的概率：$Init_{0...m}$

4. 转移概率(隐状态之间的转移概率)：$TransP(m*m 矩阵)$

5. 发射概率(隐状态输出显状态的概率):$EmitP(m*K_i)$

   $K_i$ :指的是第i个隐状态的显状态的个数

示意图:

 <img src="C:\Users\Administrator\Desktop\ms\ml\img\HMM_HiddenStateTrans.jpg" alt="HMM_HiddenStateTrans" style="zoom:80%;" />

HMM的三个问题:

1. 已知所有的模型参数$\lambda$，求$P(O|\lambda)$   ：**概率计算问题**：前向算法--**全概率分解**
2. 已知观测序列、转移概率、发射概率 如何求得使得观测序列出现的概率最大的隐状态序列**S**。  ：**预测问题**
3. 如何调整模型，使用极大似然法使得$P(O|\lambda)$最大    ：**学习问题**

### 预测问题

预测问题：**decoding**问题 --**viterbi 算法**

转化为动态规划问题--已知转换矩阵求**可以得出观测序列的最大概率**的问题

设隐序列长度为3，隐状态取值为3种

| $state_1$ | $state_2$ | $state_3$ | 隐状态 |
| --------- | --------- | --------- | ------ |
| $x_1$     | $x_1$     | $x_1$     |        |
| $x_2$     | $x_2$     | $x_2$     |        |
| $x_3$     | $x_3$     | $x_3$     |        |
| $o_1$     | $o_2$     | $o_3$     | 显状态 |

  一共有$3^3=27种选择$

求$state_1$到$state_3$ && 显状态为（$o_1$$o_2$$o_3$） 的最大概率路径

动态规划问题定义：

1. 状态转换方程：dp[i+1]= Math.max(dp[i]*$transP_{i,j}$ *$Emit_j(o_{t+1})$)
2. 定义dp数组  ：dp[i]：隐状态从0-i的最大概率

viterbi 算法实现：

```java
/**
* 求解HMM模型
* @param obs 观测序列
* @param states 隐状态
* @param start_p 初始概率（隐状态）
* @param trans_p 转移概率（隐状态）
* @param emit_p 发射概率 （隐状态表现为显状态的概率）
* @return 最可能的序列
*/

int[] compute(int[] obs, int[] states, double[] start_p, double[][] trans_p, double[][] emit_p){
   double[][] V = new double[obs.length][states.length];
   int[][] path = new int[states.length][obs.length];
   for (int y : states) //初始化第一个隐状态
    {
        V[0][y] = start_p[y] * emit_p[y][obs[0]];
        path[y][0] = y;
    }
     for (int t = 1; t < obs.length; ++t){ //遍历所有的观测序列
          int[][] newpath = new int[states.length][obs.length]; //存储新的路径
          for (int y : states){  //当前对应的隐状态
               double prob = -1; //直到t时刻，出现观测序列的隐状态的最大概率
               int state;
               for (int y0 : states){//当前的前一个隐状态
                   double nprob = V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]];
                   //出现观测状态obs[t]的的最大概率     V[t - 1][y0]：前一个隐状态
                    if (nprob > prob){
                          prob = nprob;
                          state = y0;
                          V[t][y] = prob;
                          System.arraycopy(path[state], 0, newpath[y], 0, t);
                          newpath[y][t] = y;
                    }
               }
          }
         path = newpath;//转换
     }
    double prob = -1;
    int state = 0; //最后一个隐状态
    for (int y : states)
    {
        if (V[obs.length - 1][y] > prob)
        {
            prob = V[obs.length - 1][y];
            state = y;
        }
    }

    return path[state];
}
```



### 学习问题：



HMM的条件：隐变量：$y_{1....t}$，观测变量：$x_{1....t}$

1. 其次马尔可夫性：一个状态只与前一个状态有关：$P(y_t|y_{1:(t-1)},x_{1:t-1})$=$P(y_t|y_{t-1})$
2. **观测独立**: $P(x_t|y_{1:(t)},x_{1:t-1})$=$P(x_t|y_t)$

GMM：高斯混合模型

HMM 是GMM的时序上的拓展



HMM（生成模型）+MEM(最大熵模型)==》MEMM（判别式模型）

MEMM：打破了HMM的观测独立的假设，label bias problem



HMM在中文分词中的应用：

   **由字构词**的分词算法：每一个词被分为**B、M、E** ,单独成词**S**

