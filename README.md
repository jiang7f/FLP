# FLP

## 一 库文件安装方法

```cmd
python -m pip install -r requirements.txt
```
## 二 注意

1. 项目版本:qiskit-1.0.2  
2. 可能是python的e的指数矩阵计算精度问题, "2_construction_method_No_optimized" 对酉矩阵判断偶尔会报错 (待解决):
   ``` cmd
   ValueError: Input matrix is not unitary.
   ```  
   for循环多层同时执行较容易报错, 每层测试单独执行报错概率较低  
   备选解决方案: 奇异值分解重构 (是否影响准确性?)
3. 文件名中的一些约定  
   - 文件名信息顺序: 问题类型_问题规模_测试参数_参数测试集
   - n m n: 如 2m2, 表示 two multiplied by two, 未标注则默认问题规模为2*2
   - n to n: 如 1to10, 表示遍历测试参数从1到10

## 三 项目结构

- 1_QAOA
  - 1.1_FLP_choose_part_constraint
  - 1.2_FLP_test_influence_of_parameter(penalty_depth_param)
- 2_QAOA+
  - 2.1_only_XYModel
  - 2.2_construction_method_No_optimized
  - 2.3_construction_method_with_optimized
- 3_COA (out-sync)
- 9_TEST
- z_library

### 1_QAOA (Quantum Approximate Optimization Algorithm)

#### 1.1_FLP_choose_part_constraint

测试仅仅选择部分约束的效果

#### 1.2_FLP_FLP_test_influence_of_parameter(penalty_depth_param)

测试参数（penalty惩罚系数、depth电路层数、parameter门初始参数）对QAOA解决FLP问题的影响

### 2_QAOA+ (Quantum Alternating Operator Ansatz)

#### 2.1_only_XYModel

只使用了XY-Model来约束 $∑σ=c$ 约束

#### 2.2_construction_method_No_optimized

通过构造对易哈密顿量施加约束, 未优化版 (速度较慢)  
命名规则:flp_initialstate_depth.ipynb (flp_初态选择_电路深度)

#### 2.3_construction_method_with_optimized
1_gnrt_u.ipynb:  
解析解求u. **O(nm)复杂度直接输出线性方程组的解**  
2_gnrt_hd_by_u.ipynb  
分析得, 构造Hd时, 调换张量积和∑顺序, 把所有∑项分为独立的门以此作用在电路上. 所有identity门可忽略, 只需要施加非单位门于对应量子位即可, Hdi可根据u二进制编码值O(1)复杂度推导.

### 9_TEST (测试部分组件)

### z_library (自己写的库函数)

#### extension.py

- output_to_file_init: 重定向输出流到文件, 目录为当前目录的"===output==="文件夹, 会记录运行时间和程序pid, 文件名为"文件名_pid_time.out". 用于断开ssh远程连接不挂起程序 (后台继续执行程序)  
- output_to_file_reset: 恢复重定向

运行程序指令:

```cmd
nohup xxx/python xxx.py &
```

标准输出流会输出到当前运行目录的 "nohup.out"

#### linear_system.py

- set_print_form: 设置numpy输出格式
- to_row_echelon_form: 把矩阵转换成行阶梯矩阵
- remove_zero_rows: 去除底部全0行
- find_free_variables: 返回主元和自由变量索引
- find_basic_solution: 求基础解析
- gnrt_cstt: 生成FLP问题约束矩阵

## 四 其他

关于迁移

```cmd
pipx run flake8-qiskit-migration ./
```