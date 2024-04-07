# FLP

## 一 库文件安装方法

python -m pip install -r requirements.txt

## 二 项目结构
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

1. gnrt_v.ipynb: **O(nm)复杂度直接输出线性方程组的解**

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

#### 4.2 linear_system.py

- set_print_form: 设置numpy输出格式
- to_row_echelon_form: 把矩阵转换成行阶梯矩阵
- remove_zero_rows: 去除底部全0行
- find_free_variables: 返回主元和自由变量索引
- find_basic_solution: 求基础解析
- gnrt_cstt: 生成FLP问题约束矩阵
