# FLP

## 一 库文件安装方法

```cmd
python -m pip install -r requirements.txt
```

## 二 项目结构

- QAOA
  - FLP_choose_part_constraint
  - FLP_test_influence_of_parameter(penalty_depth_param)
- QAOA+
  - checkout
  - larger
  - only_XYModel
- TEST
- zlibrary
  - extension.py
  - linear_system.py

### 1. QAOA (Quantum Approximate Optimization Algorithm)

1.1 FLP_choose_part_constraint

测试仅仅选择部分约束的效果

#### 1.2 FLP_FLP_test_influence_of_parameter(penalty_depth_param)

测试参数（penalty惩罚系数、depth电路层数、parameter门初始参数）对QAOA解决FLP问题的影响

### 2. QAOA+ (Quantum Alternating Operator Ansatz)

#### 2.1 checkout

验证约束哈密顿量效果  
文件名规则:flp_depth.ipynb

#### 2.2 larger

对于更大规模问题时间复杂度优化 (**在写**)

1. 0_gnrt_u.ipynb: **O(nm)复杂度直接输出线性方程组的解**
2. 1_gnrt_hd_by_u.ipynb  


> 首先设 $u= \{-1,0,1\}^{\otimes n}$ ,则Hd的非零元行坐标的取法为：
> 1. 将 -1 取 0 ，0 取为 (0,1), 1 取 1 ，逐个相乘得到非零元的位置。
> 例如 $u = (-1,0,-1,0,1)$ 则 非零元位置为 $0\cdot \begin{bmatrix}0\\1 \end{bmatrix}\cdot0 \cdot \begin{bmatrix}0\\1 \end{bmatrix} \cdot1$
> 即 $\begin{bmatrix}00001\\00011\\01001\\01011\end{bmatrix}$
> 2. 将-1 取 1，1取 0 ，0还是取0，1 ，得到对称的位置。
> 同样的例子，也就是将u的-1和1 出现的位置取反，得到 $1\cdot \begin{bmatrix}0\\1 \end{bmatrix}\cdot1 \cdot \begin{bmatrix}0\\1 \end{bmatrix} \cdot0=\begin{bmatrix}10100\\10110\\11100\\11110\end{bmatrix}$
> 上面两个例子综合得到非零元行坐标 $\begin{bmatrix}00001\\00011\\01001\\01011\end{bmatrix}$，$\begin{bmatrix}10100\\10110\\11100\\11110\end{bmatrix}$
> 那么列坐标就是$\begin{bmatrix}10100\\10110\\11100\\11110\end{bmatrix}$，$\begin{bmatrix}00001\\00011\\01001\\01011\end{bmatrix}$
> 即交换一下两个对称的位置组合。  因此上面的Hd的非零元坐标为(00001,10100),(10100,00001)  ,  (00011,10110),(10110,00011)  等等。

#### 2.3 only_XYModel

只使用了XY-Model来约束 $∑σ=c$ 约束

### 3. TEST (测试部分组件)

### 4.zlibrary (自己写的库函数)

#### 4.1 extension.py

- output_to_file_init: 重定向输出流到文件, 目录为当前目录的"===output==="文件夹, 会记录运行时间和程序pid, 文件名为"文件名_pid_time.out". 用于断开ssh远程连接不挂起程序 (后台继续执行程序)  
- output_to_file_reset: 恢复重定向

运行程序指令:

```cmd
nohup xxx/python xxx.py &
```

标准输出流会输出到当前运行目录的 "nohup.out"

#### 4.2 linear_system.py

- set_print_form: 设置numpy输出格式
- to_row_echelon_form: 把矩阵转换成行阶梯矩阵
- remove_zero_rows: 去除底部全0行
- find_free_variables: 返回主元和自由变量索引
- find_basic_solution: 求基础解析
- gnrt_cstt: 生成FLP问题约束矩阵

## others

关于迁移

```cmd
pipx run flake8-qiskit-migration ./
```