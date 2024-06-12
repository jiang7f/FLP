# constrained binary optimization

## 1. 设施选址问题 Facility Location Problem (FLP)

**变量**：

- 假设有 $m$ 个需求点, $n$ 个设施
- $x_j \text{ for } j= 1, \cdots, n$, 当设施 $j$ 开设时 $x_j = 1$, 否则为 $0$
- $y_{ij} \text{ for } i = 1, \cdots, m \text{ and } j = 1, \cdots, n$, 当需求点 $i$ 被分配给位置 $j$ 时 $y_{ij} = 1$, 否则为 $0$

**目标函数**：

- $min \sum_{i=1}^m \sum_{j=1}^n c_{ij} y_{ij} + \sum_{j=1}^n f_j x_j$  
其中 $f_j$ 是在位置 $j$ 建设设施的成本, $c_{ij}$ 是将客户 $i$ 分配给设施 $j$ 的成本

**约束**：

- $\sum_{j=1}^n y_{ij} = 1 \text{ for all }  i=1,2, \cdots, m$
- $y_{ij} \leq x_j \text{ for all } i=1, \cdots, m\text{ and } j=1, \cdots, n$
- $y_{i j}, x_j \in\{0,1\}$

**等式约束**:

- $\sum_{j=1}^n y_{ij} = 1 \text{ for all }  i=1,2, \cdots, m$
- $y_{i j}+z_{i j}-x_j=0 \text{ for all } i=1, \cdots, m\text{ and } j=1, \cdots, n$
- $z_{i j}, y_{i j}, x_j \in\{0,1\}$

## 2. 图着色问题 Graph Coloring Problem (GCP)

**变量**:

- 假设有 m 块图, n 种颜色, p个相邻图对
- $x_{ij} \text{ for } i = 1, \cdots, m \text{ and } j = 1, \cdots, n$, 当图 $i$ 被分配颜色 $j$ 时 $x_{ij} = 1$, 否则为 $0$

**目标函数**:

- $\min n$

**约束**:

- $\sum_{j = 1}^n x_{ij} = 1 \text{ for all } i = 1, \cdots, m$
- $x_{a_kj} + x_{b_kj} \leq 1$ for all pair of adjacent graphs $(a_k, b_k), k = 1, \cdots, p$ and $j = 1, \cdots, n$
- $x_{ij} \in\{0,1\}$

**等式约束**:

- $\sum_{j = 1}^n x_{ij} = 1 \text{ for all } i = 1, \cdots, m$
- $x_{a_kj} + x_{b_kj} - y_{jk} = 0$ for all pair of adjacent graphs $(a_k, b_k), k = 1, \cdots, p$ and $j = 1, \cdots, n$
- $x_{ij}, y_{jk} \in\{0,1\}$

## 3. 最大团问题 Maximum Clique Problem (MCP)

**变量**:

- 假设有 m 个顶点, p个不相连点对
- $x_i \text{ for } i = 1, \cdots, m$, 当顶点 $i$ 在团中时 $x_i = 1$, 否则为 $0$  

**目标函数**:

- $\max \sum_{i=1}^m x_i$  

**约束**:

- $x_{a_k} + x_{b_k} \leq 1$ for all pair of unconnected vertices $(a_k, b_k), k = 1, \cdots, p$
- $x_{i} \in\{0,1\}$

**等式约束**:

- $x_{a_k} + x_{b_k} - y_{k} = 0$ for all pair of unconnected vertices $(a_k, b_k), k = 1, \cdots, p$
- $x_{i}, y_{k} \in\{0,1\}$  

## 4. 集合覆盖问题 Set Cover Problem (SCP)

**变量**:

- 假设有 m 个集合, n 个元素
- $x_i \text{ for } i = 1, \cdots, m$, 当选择集合 $i$ 时 $x_i = 1$, 否则为 $0$

**目标函数**:

- $\min \sum_{i=1}^m x_i$

**约束**:

- $\sum_{i: e_j \in S_i} x_i \geq 1$ for all $j = 1, \cdots, n$.  
  即每个元素 $e_j$ 至少被一个集合覆盖
- $x_{i} \in\{0,1\}$  

**等式约束**:

- 没想好怎么表述元素属于多少个集合, 写了len
- $\sum_{i: e_j \in S_i} x_i - \sum_{k=1}^{len(e_j \in S_i) - 1}y_{jk}= 1$ for all $j = 1, \cdots, n$. 
- $x_{i}, y_{jk} \in\{0,1\}$

## 5. 旅行商问题 Traveling Salesman Problem (TSP)

**变量**：
- $x_{ij} = 1 如果旅行路线从城市  i  直接到城市  j ，否则为 0$

**目标函数**：
- $\min \sum_{i=1}^n \sum_{j=1, j \neq i}^n c_{ij} x_{ij} ，其中  c_{ij}  是从城市  i  到城市  j  的旅行成本$

**约束**：
- $\sum_{j=1, j \neq i}^n x_{ij} = 1  对每个城市 i$  
- $\sum_{i=1, i \neq j}^n x_{ij} = 1  对每个城市 j$  
- 防止出现子回路的额外约束，例如使用MTZ约束  
![`alt text`](TSP_add.png)