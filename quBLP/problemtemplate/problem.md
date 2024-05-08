# constrained binary optimization

### 1. 设施选址问题

**变量**：
- $x_i  = 1 如果在位置  i  建设设施，否则为 0$
- $y_{ij}  = 1 如果客户  j  被分配给位置  i  的设施，否则为 0$

**目标函数**：
- $min \sum_{i=1}^m \sum_{j=1}^n c_{ij} y_{ij} + \sum_{j=1}^n f_j x_j $  
$其中  f_j  是在位置  j  建设设施的成本， c_{ij}  是将客户  i  分配给设施  j  的成本$

**约束**：
- $\sum_{j=1}^n y_{ij} = 1, \quad i=1,2, \cdots, m$
- $y_{ij} \leq x_j, \quad i=1, \cdots, m, j=1, \cdots, 
n$
- $y_{i j}, x_j \in\{0,1\}$

**等式约束**
- $\sum_{j=1}^n y_{ij} = 1, \quad i=1,2, \cdots, m$
- $y_{i j}+z_{i j}-x_j=0, \quad i=1, \cdots, m, j=1, \cdots, n$
- $z_{i j}, y_{i j}, x_j \in\{0,1\} $

## 2. 旅行商问题（TSP）

**变量**：
- $x_{ij} = 1 如果旅行路线从城市  i  直接到城市  j ，否则为 0$

**目标函数**：
- $\min \sum_{i=1}^n \sum_{j=1, j \neq i}^n c_{ij} x_{ij} ，其中  c_{ij}  是从城市  i  到城市  j  的旅行成本$

**约束**：
- $\sum_{j=1, j \neq i}^n x_{ij} = 1  对每个城市 i$  
- $\sum_{i=1, i \neq j}^n x_{ij} = 1  对每个城市 j$  
- 防止出现子回路的额外约束，例如使用MTZ约束  
![`alt text`](image.png)

## 3. 集合覆盖问题

**变量**：
- $x_j  = 1 如果选择集合  j ，否则为 0$

**目标函数**：
- $\min \sum_{j=1}^m x_j$

**约束**：
- $\sum_{j: e_i \in S_j} x_j \geq 1  对每个元素  e_i ，其中  S_j  是包含元素  e_i  的集合$

### 4. 最大团问题

**变量**：
- $x_i  = 1 如果顶点  i  在团中，否则为 0$  

**目标函数**：
- $\max \sum_{i=1}^n x_i$  

**约束**：
- $x_i + x_j \leq 1  对于所有不相连的顶点对  (i, j)$

### 5. 图着色问题

**变量**：
-  $x_{i,k}  = 1 如果顶点  i  被分配颜色  k ，否则为 0$

**目标函数**：
- $\min  K $

**约束**：
- $\sum_{k=1}^K x_{i,k} = 1  对每个顶点  i$
- $x_{i,k} + x_{j,k} \leq 1  对每对相邻顶点  (i, j)  和每种颜色  k $
