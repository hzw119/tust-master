http://yanyiwu.com/work/2012/09/10/SVD-application-in-recsys.html

###### SVD 分解：

  对于任意一个M*N 的矩阵A,可以被写成三个矩阵的乘积：

1. U: M*M

2. S: M*N

3. V: N*N

   A=USV

![svd-recsys](http://yanyiwu.com/img/svd-recsys-p1.png)

当一个Bob[5,5,0,0,0,5] 需要个性化推荐时：

我们的思路首先是利用新用户的评分向量找出该用户的相似用户：

  $Bob_{2D}$=$Bob^{T}*U*S_{k}^{-1}$

得到Bob用户对应的坐标

然后找出和Bob最相似的用户:使用**余弦相似度**计算
$$
\text{cosine_sim}(u, v) = \frac{
\sum\limits_{i \in I_{uv}} r_{ui} \cdot r_{vi}}
{\sqrt{\sum\limits_{i \in I_{uv}} r_{ui}^2} \cdot
\sqrt{\sum\limits_{i \in I_{uv}} r_{vi}^2}
}
$$
计算**向量相似度**的方法:

然后采用推荐策略：对相似用户没有看过的评分高的电影进行推荐



notes:直接的SVD费时费力，可以通过学习的方法进行矩阵分解

 V=U*M

  V: m*n

  U: m*k

  M: k*n

定义损失函数:loss func
$$
E = \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{m}I_{ij}(V_{ij} - p(U_i,M_j))^2+\frac{k_u}{2}\sum_{i=1}^{n}\lVert U_i \rVert^2 + \frac{k_m}{2}\sum_{j=1}^{m}\lVert M_j \rVert^2 
$$
使用梯度下降法最小化loss func：
$$
-\nabla _{U} =-\frac{\partial E}{\partial U_i} = \sum_{j=1}^{M}I_{ij}((V_{ij}-p(U_i,M_j))M_j) - k_uU_i \tag{2}
$$

$$
-\nabla _{M} =-\frac{\partial E}{\partial M_j} = \sum_{i=1}^{n}I_{ij}((V_{ij}-p(U_i,M_j))U_i) - k_mM_j \tag{3}
$$

**improvements(adding bias):**

reasons: typical collaborative filtering data exhibits large systematic tendencies for some users to give higher ratings than others, and for some items to receive higher ratings than others.
$$
\hat{r}_{ui} = \mu + b_i + b_u + \mathbf{q}_i^T\mathbf{p}_u \tag{4}
$$
$\mu$:训练集中所有记录的评分的全局平均数，在不同网站中，因为网站定位和销售的物品不同

$b_i$:用户偏置

$b_u$:物品偏置

**new loss function:**
$$
E = \sum_{(u,i)\in \mathcal{k}}(r_{ui}-\mu - b_i - b_u - \mathbf{q}_i^T\mathbf{p}_u)^2 + \lambda (\lVert p_u \rVert^2 + \lVert q_i \rVert^2 + b_u^2 + b_i^2) \tag{5}
$$
梯度下降法：
$$
b_u\leftarrow b_u + \gamma (e_{ui} - \lambda b_u) \tag{6}
$$
