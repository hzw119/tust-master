# 最大似然估计（MLE）

数学表达：$\theta_{MLE}$=$argmax_{\theta} logp(x|\theta)$

本质：知道结果，反推条件$\theta$   已知**模型和样本**-->推导模型参数 

解决的问题：概率分布**只有一个或者知道**结果是通过哪个**概率分布**实现的

# EM

目的：**解决具有隐变量的混合模型**的参数估计

数学表达(如何证明)： 

数学表达解释:![Em](C:\Users\Administrator\Desktop\ms\ml\img\Em.svg)

   $\theta$:模型参数，Z:隐状态，X：是观测数据

解决的问题：样本的分布不明确

隐变量：

一般使用Y表示观测到的随机变量的数据，Z表示**隐随机变量**（我们观测不到Y是由哪个概率分布得出的）的数据，Y+Z 被称作完全数据(p(x,z|$\theta$ ))。

EM算法的步骤：

1. 初始化$\theta$.

2. (E步)根据$\theta$和观测数据Y估计Z；
3. (M步）根据第二步计算出的Z更新$\theta$（使用最大似然估计,,使Z出现的可能性最大）；

五枚不同种类的硬币各抛五次，得出观测结果Y，

比如只有两个种类：A、B,需要计算的是PA、PB为A、B出现正面的几率

1.初始化PA、PB

2.(E步)根据PA、PB和观测序列得出五枚硬币最有可能属于的种类,得到Z；

3.(M步) 根据得出的Z更新PA、PB。



EM算法的步骤的数学化表达

1. E step: 计算在p(z|x,$\theta^{t}$) 条件下log(p(x,z|$\theta^{t} $))的期望，**条件期望**

   ![eStep](C:\Users\Administrator\Desktop\ms\ml\img\eStep.svg)

2. M step:最大化此期望以更新$\theta $

​            ![mstep](C:\Users\Administrator\Desktop\ms\ml\img\mstep.svg)





jensen 不等式：

​    如果某函数f(x)是凹函数，则 f(E(x))<=E(f(x))，当x为常量时，等号成立

参考博客：

https://zhuanlan.zhihu.com/p/40991784





