1. 输入n和a计算a+aa+aaa+a..(n个a)

```java
public int  nOfa(int n,int a){
    int sum=0;
    for(int i=0,j=n;i<n;i++,j--){
        sum+=a*j*Math.pow(10,i);
    }
    return sum;
}
```

   2. 计算1！＋2！＋3！＋……＋100！

      ```java
      public int sum(){
          int sum=0;
          for(int i=1;i<=100;i++){//计算1-100的阶乘和
              sum+=jiecheng(i); 
          }
          return sum;
      }
      public int jiecheng(int a){ //求解a！
          if(a==1) return 1;
          return a*jiecheng(a-1);
      }
      ```

      

3. 输入一个正整数，判断是不是质数

   ```java
    for(i=2;i<num/2;i++)
    {
        if(num%i==0)
        {
           break;
        }
    }
   ```

4. 将一个链表中的前k个连续结点**逆置**

   

5. 把一串字符中连续的'abc'换成'ab'

   ```java
   public String replaceAll(String s){
       char[] chars=s.toCharArray();
       StringBuilder s_temp=new StringBuilder();
       int num=0;// 'abc'的个数
       for(int i=0;i<s.length();i++){
           if(i-2>=0&&chars[i]=='c'){
               if(chars[i-1]=='b'&&chars[i-2]=='a'){
                   continue;
               }
           }
           s_temp.append(chars[i]);
       }
       return s_temp.toString();
   }
   ```

   

6. 给定一个整型数组，和一个整型的sum，求数组中任意两数之和等于sum的不同组合的个数。

   ```java
   public  int twoSum(int[] nums, int target) {
       int result=0; //返回的组合数
       HashMap<Integer,Integer> hash = new HashMap<Integer,Integer>();
       for(int i = 0; i < nums.length; i++){
           if(hash.containsKey(target-nums[i])){
               result++;
           }
           // 将数据存入 key为补数 ，value为下标
           hash.put(nums[i],i); //可以去重
       }
       return result;
   }
   ```

   

7. 快速排序：

   ```java
   public void quicksort(int [] nums){
       quicksorthelper(nums,0,nums.length);
   }
   public void quicksorthelper(int [] nums,int start,int end){
       int privot=nums[start];
       int i=start;
       int j=end;
       if (i>=j||i>nums.length-1) return;
       while(i<j){ //i==j 时循环终止
           while(i<j&&nums[j]<privot){
               j--;
           }
           nums[i]=nums[j];
           while(i<j&&nums[i]>=privot){
               i++;
           }
           nums[j]=nums[i];
       }
       nums[i]=privot;
       quicksorthelper(nums,start,i-1); //递归左半部分
       quicksorthelper(nums,i+1,end);   //递归右半部分
   }
   ```

   

8. 找出二叉搜索树中出现最频繁的节点

   ```java
   class TreeNode{
      TreeNode left;
      TreeNOde right;
      int val;
      TreeNode(int val){
         val=val;
      }
   }
   int maxFreqVal;
   int maxFreqCount;
   int curFreqVal = -1;//不一定要为-1 只要未在树中出现的值就行
   int curFreqCount = -1;
   void inorder(TreeNode node){
       if(node==NULL) return ;
       inorder(node.left);
       if(curFreqVal!=node.val){ //当遍历到时不是当前值时，更新
           curFreqCount=1;
           curFreqVal=node.val;
       }else{  //当遍历到时是当前值时，更新
           curFreqCount++;
           if(curFreqCount>maxFreqCount){
               maxFreqCount=curFreqCount;
               maxFreqVal = curFreqVal;
           }
           
       }
       inorder(node.right);
   }
   int getFreq(TreeNode  root){
       if(root ==NULL) return -1;
       inorder(root);
       return maxFreqVal;
   }
   ```


9. 给定一棵树的两个节点，求最近的公共祖先

   ```java
   1.树节点有指向父节点的指针
       转化为求链表的公共序列的首节点
   2.树节点无指向父节点的指针
       1.后序遍历
       2.递归查询，当一节点a是两个节点b,c的最近公共祖先时，表明b,c 分别在a的左右子树
   //无堆栈
   TreeNode GetLastCommonAncestor(TreeNode root, TreeNode node1, TreeNode node2){
       if(root==null||node1==null||node2==null){
           return null; //查询失败
       }
       if(node1==root||node2==root){
           return root;  //在左右子树查询成功
       }
       TreeNode left_lca = GetLastCommonAncestor(root->left, node1, node2);
   	TreeNode right_lca = GetLastCommonAncestor(root->right, node1, node2);
       if (left_lca && right_lca)
   		return root; //左右子树查询成功则返回当前节点
       if(left_lca == NULL){  //node1、node2都不在当前节点的左子树
           return right_lca;  //node1、node2 可能在当前节点的右子树
       }else{
            return left_lca;  //node1、node2中的一个在当前节点的左子树
       }
       
   }
   ```

10. 求两个链表的公共序列

    ```java
    public ListNode findCommon(ListNode list1,ListNode list2){
        int len1=lenOflist(list1);
        int len2=lenOflist(list2);
        //末尾对齐
        for(;len1>len2;len1--){
            list1=list1.next;
        }
        for(;len2>len1;len2--){
            list2=list2.next;
        }
        //两个链表等长
        while(list1&&list2&&list1!=list2){
             list1=list1.next;
             list2=list2.next;
        }
        if(node1==node2){
            return node1;
        }else{
            return null;
        }
    }
    public int lenOflist(ListNode node){
        if(node==null) return 0;
        if(node.next==null) return 1;
        return lenOflist(node.next)+1;
    }
    ```

11. 最长公共子**序列**/**子串**

    ```java
    序列不一定连续
    public int longestCommonSubsequence(String text1, String text2) {
            int n=text1.length();
            int m=text2.length();
            int [][]dp=new int[n+1][m+1];
            for (int i = 1; i <n+1; i++) {
                for (int j = 1; j < m+1; j++) {
                    if (text1.charAt(i-1)==text2.charAt(j-1)){
                        dp[i][j]=dp[i-1][j-1]+1;
                    }else{
                        dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                    }
    
                }
            }
            return dp[n][m];
    }
    子串必须连续
    public int longestCommonSubStr(String text1, String text2) {
            int n=text1.length();
            int m=text2.length();
            int result=0;
            int [][]dp=new int[n+1][m+1]; 
            //dp[i][j]表示以text1.charAt(i),text2.charAt(j)为截尾的最长公共子串的长度
            for (int i = 1; i <n+1; i++) {
                for (int j = 1; j < m+1; j++) {
                    if (text1.charAt(i-1)==text2.charAt(j-1)){
                        dp[i][j]=dp[i-1][j-1]+1;
                        reshult=Math.max(result,dp[i][j])
                    }
    
                }
            }
            return result;
    }
    
    
    
    ```

12. 最佳加法表达式

    ```java
    dp[i][j] 表示i个数字j个加号的最小和
    j=0 dp[i][j]=i个数字构成的整数
    i<j+1 dp[i][j]=正无穷 此时表达式是非法的
    dp[i][j]=min{dp[i-1][q]+num(q+1,i)}(q=j....i-1)
    加号放在数字串q位右边的位置，q的范围是(j...i-1)
        
    public int  bestStrategy(String s,int n){
        
    }
    ```

13. 一个**数组**里两两一对数怎么找出单独的一个

    ```java
    xor异或：相应位相同为0，不同为1
    a^b^b=a
    b^a^b=a
    0^n=n
    public int findOne(int nums){
        int t=0; 
        for(int i=0;i<nums.length();i++){
           t^=nums[i];   
        }
        return t;
    }
    异或满足交换律
    ```

    

14. 大数相加

    ```cpp
    char* a="11111111111111111111111111";
    char* b="91111111111111111111111111111111111";
    
    int aLen=strlen(a);
    int bLen=strlen(b);
    int maxlen=alen>blen?alen:blen; //获得最长的数据长度
    int[] aArr=new int[maxlen];
    int[] bArr=new int[maxlen];
    int[] sumArr=new int[maxlen]; //+1 以防进位
    for(int i=maxlen-1;i>=0;i--){
        if(alen<=0){ 
            alen=0;
            aArr[i]=0; //尾部对齐，不够补0 
        }else{
            aArr[i]=a[--alen]-'0';
        }
    }
    对b相同操作
    //实现加法
    int jin =0;
    for(int i=maxlen-1;i>=0;i--){
        sumArr[i]=(aArr[i]+bArr[k]+jin)%10;
        jin=(aArr[k]+bArr[k]+jin)/10;
    }
    if(jin==1){
        //最前的一位是1
    }
    ```


15. 大数相乘

    ```java
public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] pos = new int[m + n];
        for(int i = m - 1; i >= 0; i--) {
            for(int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0'); 
                int p1 = i + j, p2 = i + j + 1;  //p1 为下位 p2 为上位
                int sum = mul + pos[p2];
                pos[p1] += sum / 10;
                pos[p2] = (sum) % 10;
           }
        }  
        StringBuilder sb = new StringBuilder();
        for(int p : pos) if(!(sb.length() == 0 && p == 0)) sb.append(p);
        return sb.length() == 0 ? "0" : sb.toString();
        
    }
    ```

​    


16. 常见的排序算法

    1) 插入排序

    ​        （1）直接插入排序：稳  O(n^2)

    ​        （2）折半插入排序：稳  O(n^2)

    ​        （3）希尔排序：缩小增量排序。不稳  O(n^(1.3—2))

    2) 交换排序

    ​        （1）冒泡排序：稳  O(n^2)

    ​        （2）快排：不稳 

    ​                   最佳：**时间复杂度O(nlogn)。** **空间复杂度：递归深度O(logn)**

    ​                   最坏： 时间复杂度O(n^2)  空间复杂度O(n)

    ​                   快排的优化方法：

    ​                        1）随机基准

    ​                        2）n数取中

    ​                        优化1：序列长度达到一定大小时，使用插入排序

    ​                        优化2：聚集元素

​     3)选择排序

​               （1）简单选择排序：

​               （2）堆排序：时间复杂度O(nlogn)  空间复杂度O(n)

​    4）归并排序

​    5）基数排序

​    6)   Hash排序，空间换时间，时间负责度取决于最大元素

16. **汉诺塔的故事**

```cpp
    void hanota(vector<int>& A, vector<int>& B, vector<int>& C) {
        int n=A.size();
        move(n,A,B,C);
    }
    void move(int n, vector<int>& A, vector<int>& B, vector<int>& C){
        if(n==1){
            C.push_back(A.back()); //最后一次移动中，最小的一个在A上边
            A.pop_back();
            return;
        }
        move(n-1, A, C, B);    // 将A上面n-1个通过C移到B
        C.push_back(A.back()); //此时最大的在A底下，移动到C中
        A.pop_back();          //A退出
        move(n-1, B, A, C);     // 将B上面n-1个通过空的A移到C
    }
```

17. 二叉树的最长直径(后序遍历) leetcode 543/687/124

    ```java
    二叉树的直径/二叉树两叶子节点的最远距离
    class Solution {
        int ans;
        public int diameterOfBinaryTree(TreeNode root) {
            ans = 1;
            depth(root);
            return ans - 1;
        }
        public int depth(TreeNode node) {
            if (node == null) return 0;
            int L = depth(node.left);
            int R = depth(node.right);
            ans = Math.max(ans, L+R+1);
            return Math.max(L, R) + 1; //返回单侧最长
        }
    }
    
    二叉树的节点最大和
    class Solution {
        int max_sum = Integer.MIN_VALUE;
        public int maxPathSum(TreeNode root) {
            getMax(root);
            return max_sum;
        }
        private int getMax(TreeNode r) {
            if(r==null) return 0;
            int left=Math.max(0,getMax(r.left));   //如果是小于零就排除掉
            int right=Math.max(0,getMax(r.right));  
            max_sum=Math.max(max_sum,r.val+left+right);
            return Math.max(left,right)+r.val; //返回单侧最长路径
        }
    }
    二叉树的同值最长半径
    class Solution {
        int ans;
        public int longestUnivaluePath(TreeNode root) {
            ans = 0;
            arrowLength(root);
            return ans;
        }
        public int arrowLength(TreeNode node) {
            if (node == null) return 0;
            int left = arrowLength(node.left);
            int right = arrowLength(node.right);
            int arrowLeft = 0, arrowRight = 0;
            if (node.left != null && node.left.val == node.val) {
                arrowLeft += left + 1;
            }
            if (node.right != null && node.right.val == node.val) {
                arrowRight += right + 1;
            }
            ans = Math.max(ans, arrowLeft + arrowRight);
            return Math.max(arrowLeft, arrowRight);
        }
    }
    ```

    

18. 找出两链表的相交结点

    ```java
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            if(headA == null || headB == null) return null;
            ListNode pA = headA, pB = headB;
            while(pA != pB) {
                pA = pA == null ? headB : pA.next;
                pB = pB == null ? headA : pB.next;
            }
            return pA;
        }
    
    如何判断两个链表是否相交：
        1.使用两个栈将两个链表的所有结点全部入栈，然后两个栈同时出栈，如果栈顶的元素相同，则链表相交，如果栈顶不同，则不相交
        2. 尾部对齐，求两个链表的长度，然后让长的链表先走几步，然后俩链表的长度为相同时，然后同时向下遍历，遇见第一个相等的结点。
    ```


19. 岛屿数量 **dfs/bfs**模板

    最短路径->bfs

    搜索全部的解->dfs
    
    dfs：
    
    ```java
    for (int i = 0; i <rows ; i++) {
        for (int j = 0; j <cols ; j++) {
            if (marked[i][j]==false&&grid[i][j]=='1'){
                res++;
                dfs(i,j); //向下递归
            }
        }
    }
    
    private void dfs(int i, int j) {
        marked[i][j]=true;
        for (int k = 0; k <4; k++) {
            int newX=i+directions[k][0];
            int newY=j+directions[k][1];
            if(inArea(newX,newY)&&grid[newX][newY]=='1'
               &&marked[newX][newY]==false){
                dfs(newX,newY);
            }
        }

    }

    ```
    
    bfs
    
    ```java
    for (int i = 0; i <rows ; i++) {
                for (int j = 0; j <cols ; j++) {
                    if (!marked[i][j]&&grid[i][j]=='1'){
                        res++;
                        //little trick
                        LinkedList<Integer> queue=new LinkedList<>();
                        queue.addLast(i*cols+j);
                        marked[i][j] = true;
                        while (!queue.isEmpty()){
                            int cur=queue.removeLast();
                            int curX = cur / cols;
                            int curY = cur % cols;
                            for (int k = 0; k <4 ; k++) {
                                int newX = curX + directions[k][0];
                                int newY = curY + directions[k][1];
                                if (inArea(newX, newY) && grid[newX][newY] == '1' && !marked[newX][newY]) {
                                    queue.addLast(newX * cols + newY);
                                    marked[newX][newY] = true;
                                }
                            }
                        }
                    }
                }
            }
    ```


20. 回文

    ```java
    验证回文
    public boolean isPalindrome(String s,int i,int j){
        while(i<j){
            if(s.charAt(i++)!=s.charAt(j--))
                return false;
        }
        return true;
    }
    
    leetcode 5 最长回文子串==>s与s的reverse后的s`的最长公共子串+判断逆转前的坐标和逆转后的坐标是否对应
    
    public String longestPalindrome(String s) {
    
    }
    
    ```

    

21. 01背包

```java
    public int hightestValue(int [] weight,int [] value, int capCity){
        int result=0;
        int [][]dp=new int[weight.length+1][capCity+1];
        //dp[i][j] i重量&& j 容量的 最大value值
        for (int i = 0; i < weight.length+1; i++) {
            dp[i][0]=0;
        }
        for (int i = 0; i <capCity+1 ; i++) {
            dp[0][i]=0;
        }
        for (int i = 1; i <capCity ; i++) {  //容量
            for (int j = 1; j <weight.length ; j++) {  //物品
                if (i-weight[j]>=0 &&dp[j-1][i-weight[j]]+value[j] > dp[j-1][i]){  //有剩余容量且装入后价值增加
                    dp[j][i]=dp[j-1][i-weight[j]]+value[j];
                }
                dp[j][i]=dp[j-1][i];   //无剩余容量或者装入后价值不增加
            }
        }
        return dp[weight.length][capCity];


    }
```



22. Kruskal、Prim

    最小生成树：在连通图的所有生成树中，所有边的代价和最小的生成树

    Kruskal算法：**加边法**,每**迭代一次**就选择一条满足条件的最小代价边，加入最小生成树的边集合里。选n-1条边。

    Prim算法：**加点法**，在**未加入的点**中选出一个到已选点中所组成的树的最短距离的点。

23. djstrala  、Floyd算法

    

24. 堆和栈的区别

    1. 程序内存方面：表示两种内存管理方式
    2. 数据结构方面：表示两种常用的数据结构

    程序内存方面：

    栈由操作**系统自动分配释放**，用于存放函数的参数值等参数。

    堆由开发**人员分配和释放**

    栈、堆空间的分配方式不同

