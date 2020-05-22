# 1. Loss function

1. L2 norm loss

2. L1 norm loss

3. Pseudo-Huber loss

4. Hinge loss（SVM）

5. Cross-entropy loss：常用于分类问题，描述了**两个概率分布之间**的距离，当交叉熵越小说明二者之间越接近。

   classification error

   为什么分类常用Cross-entropy loss？

   $H_{y‘}(y) := -\sum_iy_i'log(y_i)$

   Cross Entropy Loss 计算 loss，就还是一个**凸优化问题**

6. Sigmoid cross entropy loss 

# 2. Optimizer

1. AdagradOptimizer():自动变更学习速率，只是需要设定一个全局的学习速率$\alpha$

   实际上的学习速率是与参数的梯度的模之和的开方成反比的。

   $\alpha_{n}$=$\frac{\alpha}{\lambda+{\sum_{i=1}^{n-1}p_i^2}}^{\frac{1}{2}}$

   Adagrad forces the gradients to zero too soon because it takes into account the whole history.

   学习率自动更改，如果梯度大，学习速率就衰减的**快**一些，如果梯度小，那么学习速率衰减的**慢**一些。

2. RMSprop

   

3. AdadeltaOptimizer()

4. MomentumOptimizer() 



激活函数：

1. 什么是激活函数？

   **上层**结点的输出和**下层**节点的输入之间的一个**函数关系**

2. **激活函数的作用**：

   如果没有激活函数，神经网络只不过是一个线性回归模型，神经网络中的激活函数的主要作用是提供网络的**非线性建模/学习**能力。

    没有的激活函数的神经网络只是一个复杂的线性模型

3. 常见 激活函数

   1.$sigmoid$函数（**软饱和**激活函数）

   在定义域内处处可导，且两侧导数趋近于0，所以当Z的值很大/小的时候，函数的梯度会非常小，在反向传播的过程中，链式导数连乘可能会导致**梯度消失**。

   一般只用于二分类的输出层

   2.$tanh$ 函数（双曲正切函数）

   $tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$

   收敛速度比sigmoid 函数快，和sigmoid一样具有**软饱和性**，从而造成梯度消失

   3.$Relu$ 函数（修正线性单元，最常用的激活函数）vs $leaky \ Relu$

   <img src="C:\Users\Administrator\Desktop\ms\ml\img\relu.webp" alt="relu" style="zoom: 50%;" />

​       Relu 在x>0时不饱和，x<0 时硬饱和，x>0 时梯度不会趋于0，缓解梯度消失的问题。

​        $leaky \ Relu$ 是针对 $Relu$ x<0 的**硬饱和问题**

​           <img src="C:\Users\Administrator\Desktop\ms\ml\img\leakyRelu.webp" alt="leakyRelu" style="zoom: 50%;" />



