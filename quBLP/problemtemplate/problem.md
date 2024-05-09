# constrained binary optimization

### 1. 设施选址问题 Facility Location Problem (FLP)

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

**等式约束**
- $\sum_{j=1}^n y_{ij} = 1 \text{ for all }  i=1,2, \cdots, m$
- $y_{i j}+z_{i j}-x_j=0 \text{ for all } i=1, \cdots, m\text{ and } j=1, \cdots, n$
- $z_{i j}, y_{i j}, x_j \in\{0,1\} $

### 2. 图着色问题 Graph Coloring Problem (GCP)

**变量**：
- 假设有 m 块图, n 种颜色
- $x_{ij} \text{ for } i = 1, \cdots, m \text{ and } j = 1, \cdots, n$, 当图 $i$ 被分配颜色 $j$ 时 $x_{ij} = 1$, 否则为 $0$

**目标函数**：
- $\min n $

**约束**：
- $\sum_{j = 1}^n x_{ij} = 1 \text{ for all } i = 1, \cdots, m$
- $x_{a,j} + x_{bj} \leq 1$ for all pair of adjacent graphs $(a, b)$ and $j = 1, \cdots, n$
- $x_{ij} \in\{0,1\} $

**等式约束**:
- $\sum_{j = 1}^n x_{ij} = 1 \text{ for all } i = 1, \cdots, m$
- $x_{aj} + x_{bj} - y_{abj} = 0$ for all pair of adjacent graphs $(a, b)$ and $j = 1, \cdots, n$
- $x_{ij}, y_{abj} \in\{0,1\} $

**pending: ab可以表示成一个k(相邻边数) ,共有km个y. 写代码的时候发现 $n^3$ qubit太多，所以按这个思路写的，公式表述待改**  
## 3. 旅行商问题 Traveling Salesman Problem (TSP)

**变量**：
- $x_{ij} = 1 如果旅行路线从城市  i  直接到城市  j ，否则为 0$

**目标函数**：
- $\min \sum_{i=1}^n \sum_{j=1, j \neq i}^n c_{ij} x_{ij} ，其中  c_{ij}  是从城市  i  到城市  j  的旅行成本$

**约束**：
- $\sum_{j=1, j \neq i}^n x_{ij} = 1  对每个城市 i$  
- $\sum_{i=1, i \neq j}^n x_{ij} = 1  对每个城市 j$  
- 防止出现子回路的额外约束，例如使用MTZ约束  
![`alt text`](TSP_add.png)

## 3. 集合覆盖问题 Set Cover Problem (SCP)

**变量**：
- $x_j  = 1 如果选择集合  j ，否则为 0$

**目标函数**：
- $\min \sum_{j=1}^m x_j$

**约束**：
- $\sum_{j: e_i \in S_j} x_j \geq 1  对每个元素  e_i ，其中  S_j  是包含元素  e_i  的集合$

### 4. 最大团问题 Maximum Clique Problem (MCP)

**变量**：
- $x_i  = 1 如果顶点  i  在团中，否则为 0$  

**目标函数**：
- $\max \sum_{i=1}^n x_i$  

**约束**：
- $x_i + x_j \leq 1  对于所有不相连的顶点对  (i, j)$

