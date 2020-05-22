1. logistics regression 为什么不搭配square Error？

   f($x_i$)=$\frac{1}{1+e^{-(w^Tx_i+b)}}$,loss = $\frac{1}{2} \sum_{i=1}^{n}(f(x_i)-y_i)^{2}$

   loss 对w 进行求导：$\sum^{n}_{i=1}(f(x_i)-y_i)f(x_i)(1-f(x_i))x_i$

   当$y_i=0$时，若$f(x_i)=0$,则loss 为0，若$f(x_i)=1$，则上式也为0，不符合loss；

   $y_i=1$时，若$f(x_i)=1$，则上式为0，符合loss；若$f(x_i)=0$，则上式也为0，不符合loss；

2. **Square Error VS. Cross Entropy Error**

   <img src="C:\Users\Administrator\Desktop\ms\ml\img\logisticRegresion.jpg" alt="logisticRegresion" style="zoom: 25%;" />

3. Discriminative(判别式) VS.Generative(生成式)

   Discriminative(判别式) =》Navie Bayes

   Generative=》Logistics Regression 

4. Multiple-class Classification

   把所有数据都划分为A类和 非A类 /B类和非B类

   分三次类：

   C1：z1=f1(w1 *x +b1)

   C2:   z2=f2(w2 *x +b2)

   C3:   z3=f3(w3 *x +b3)

   然后经过一次 softmax([z1,z2,z3])=[y1,y2,y3] 与  target[$y^{1},y^{2},y^{3}$] 做cross entropy

5. **limitation** of logistics regression

   没有解决异或问题，因为logistics regression 是**一条边界**

   solution：feature  Transformation

    **but** Not always easy to find a good transformation

   **神经单元**==》feature transformation
   
6. logistics regression vs linear regression

   逻辑回归是一种分类问题，线性回归是一种回归问题，

   逻辑回归为啥叫回归：y(0/1)=sigmoid($w^Tx+b$)= $\frac{1}{1+e^{-(w^Tx+b)}}$=>

   ​             $Log(\frac{y}{1-y})$=$w^tx+b$



## Neural Network

1. feature engineering 

2. loss func: **cross entropy**

3. mini-Batch:**we don`t  really minimize total loss**

   分批拟合

**Tips for Deep learning**





**Semi**-supervised learning

1. A set of unlabeled data
2. transductive learning（传递学习）: unlabeled data is testing data
3. inductive learning(归纳学习):unlabeled data is not the testing data.