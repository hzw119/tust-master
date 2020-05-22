SVM

1. Hard-margin SVM
2. soft-margin SVM
3. kernel SVM



不等式约束优化（**Constraint Optimization**）问题：
$$
min f(x) \qquad x \in R^{p}    \\
s.t. m_{i}(x)<=0 \ i=1.2...m   \\
    \ \ n_{j}(x)=0  \ \  j=1.2...n   \\
$$


## **SVM的求解过程：**

1. 数学模型定义：$f(w)=\frac{1}{2} ||w||^{2} s.t. y_{i}(w^{t}x+b)>=1$

2. **拉格朗日乘子法**：有约束问题转换为无约束问题

3. **对偶问题**转化：不易求解的问题转化为**易求解的对偶问题**，对偶问题涉及强**对偶**问题、弱对偶问题

   如何证明**对偶问题的强对偶性**：**KKT条件**

   **KKT** 与强对**偶问题**互为**充要**条件

   KKT条件：
   $$
   条件一：\frac{\partial loss(w,b,\lambda)}{\partial x}=0   \\
   条件二： n_{j}(x)=0                      \\
   条件三：\lambda_{i} m_{i}(x)=0
   $$
   也就是说只要证明$loss(w,b,\lambda)$满足KKT条件，则可证明原问题是一个强对偶性问题。

**kernel SVM:**

通常的步骤：先找一个kernel函数，把低维向量映射到高维向量，然后，再在高维空间中求超平面。

这样的做法有什么弊端：

1. 合适的kernel函数难以寻找
2. 当映射到高维空间中，可能出现维度灾难
3. 计算复杂，先映射再求超平面

所以引入：**kernel trick**: **对偶带来的内积**

**kernel function**:  对于任意x1，x2 $\in$  X ,存在$\phi() \ \in$ Z,   $K(x1,x2) = <\phi(x1), \phi(x2)>$

**kernel trick**: 定义一个核函数$K(x1,x2) = <\phi(x1), \phi(x2)>$, 其中$x_1$和$x_2$是低维度空间中点（在这里可以是标量，也可以是向量），$\phi(xi) $是低维度空间的点xi转化为高维度空间中的点的表示，< , > 表示向量的内积。

kernel trick 中 我们**不用关注高维度空间**的形式，在高维度中向量的内积通过低维度的点的核函数就可以计算了

**那么为什么引入内积？**

kernel trick 是针对于内积问题

Dual problem（对偶问题）：
$$
min \sum_{i}\sum_{j} \lambda_{i} \lambda_{j}y_{i}y_{j}x_{i}^{T}x_{j}+\ \sum_{i} \lambda_{i}
$$
**对偶**问题带来**内积**：$x_{i}^{T}x_{j}$转换为 $<\phi(x1), \phi(x2)>$



Hard-margin SVM

 **loss func:**
$$
min \frac{1}{2} w^Tw \qquad x \in R^{p}    \\
s.t. y_i(w^Tx_i+b)>=1
$$
Soft-margin SVM

由于数据点完全线性可分的情况很少，所以难免有分类错误的点，所以把分类错误的点考虑进来，则降低分类错误时的差值，分类错误时，损失为$1-y_i(w^Tx_i+b)$，正确时为0，所以Soft-margin SVM由分类错误带来的损失为

Max(0,$1-y_i(w^Tx_i+b)$)

**loss func:**

​    $\frac{1}{n} \sum_{i}^{n} Max(0,1-y_i(w^Tx_i+b))$+ a${\mid \mid A\mid \mid}^{2}$

   



## SVM regression

找一个最大间隔的"宽带"去包含尽可能多的样本点，然后回归线落在这个**宽带**的中间。

如图：

<img src="C:\Users\Administrator\Desktop\ms\ml\img\3.png" alt="3" style="zoom:67%;" />

loss func：

   $ \frac{1}{n}\sum_i^{n}Max(0,|y-(W^Tx+b)|-\lambda)$

tensorflow写法：

```python
loss = tf.reduce_mean(tf.maximum(0., tf.sub(tf.abs(tf.sub(model_
output, y_target)), epsilon)))

model_output:模型的输出
y_target:目标输出
epsilon：宽带的宽度的一半
```

$\lambda$ 为这个宽带宽度的一半

##  高斯kernel函数：

```python
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.add(tf.sub(dist, tf.mul(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))

```

$$
dist=\sum x_i^2 \\
sq\_dists=dist^T+(dist-2XX^T)
$$

