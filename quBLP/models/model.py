## This file defines a problem of binary constraint optimization 
from quBLP.utils import iprint, set_print_form
import numpy as np
import itertools
from typing import Iterable, List, Callable,  Union
from ..utils.linear_system import find_basic_solution, to_row_echelon_form
from ..utils.parse_expr import split_expr
from ..utils import QuickFeedbackException
from ..solvers.solver import solve
from dataclasses import dataclass, field
from .option import OptimizerOption, CircuitOption
import gurobipy as gp

class Model:
    def __init__(self) -> None:
        pass

class Expression:
    def __init__(self,expr:str) -> None:
        self.expr = expr
        pass
    def __add__(self,other):
        if isinstance(other, Expression):
            return self.expr + other.expr
        
        return self.expr + other
    def __sub__(self,other):
        return self.expr - other
    def __eq__(self,other):
        return self.expr == other
    
   
class Variable:
    def __init__(self,name:str) -> None:
        self.name = name
        self.x = 0
        pass
    def set_value(self, value):
        self.x = value
    def __eq__(self,other):
        return self.x == other
    def __add__(self,other):
        if isinstance(other,Variable):
            return self.x + other.x
        return Expression(self) + Expression(other)
    def __sub__(self,other):
        if isinstance(other,Variable):
            return self.x - other.x
        return self.x - other
    def __mul__(self,other):
        if isinstance(other,Variable):
            return self.x * other.x
        return self.x * other
    
class ConstrainedBinaryOptimization(Model):
    def __init__(self, fastsolve=False):
        """ a constrainted binary optimization problem

        Args:
            fastsolve (bool, optional): whether to use fast solve method to get the bitstring for driver hamiltonian. Defaults to False.
        statement:
            objective_func_term_list: 用于存储目标函数中的各次项
            每一个元素是一个列表，表示一个次项（一次项、二次项等）
            每个次项列表包含若干项，每项是一个包含两个元素的列表：
            第一个元素是变量序号的列表（对应该项的变量）
            第二个元素是该项的系数
            例如：
            一次项列表: [[[0], 2.5], [[1], 3.0]] 表示 2.5*x_0 + 3.0*x_1
            二次项列表: [[[0, 1], 1.5]] 表示 1.5*x_0*x_1
            完整结构：[[一次项列表], [二次项列表], ...]
        """
        set_print_form()
        self.fastsolve = fastsolve
        self._driver_bitstr = None
        self.variables = []
        # self._constraints 预留为 linear 和 nonlinear 的并集
        self._linear_constraints = None  # yeld by @property 是 for_cyclic 和 for_others 的并集
        self._constraints_classify_cyclic_others  = None # cyclic 用于存∑=x
        self.objective_func_term_list = [[], []] # 暂设目标函数最高次为2, 任意次的子类自行重载, 解释见 statement
        self.objective_func = None
        self.objective_penalty = None
        self.objective_cyclic = None
        self.objective_commute = None
        self.variables_idx = {}
        self.current_var_idx = 0
        self.variable_name_set = set()
        self.algorithm_optimization_method = 'commute'
        self.penalty_lambda = 0
        self.collapse_state = None
        self.probs = None
        self.cost_dir = 0
        self.opt_mtd = 'standard'
        pass

    def set_optimization_direction(self, dir):
        assert dir in ['min', 'max']
        self.cost_dir = 1 if dir == 'min' else -1 if dir == 'max' else None

    def find_state_probability(self, state):
        try:
            index = self.collapse_state.index(state)
        except ValueError as e:
            print('find_state_probaility', e)
            return 0
        else:
            return self.probs[index]
    
    def set_algorithm_optimization_method(self, type='commute', penalty_lambda = None):
        """
        Set the optimization method for the algorithm.

        Args:
            type (str, optional): the optimization method for the algorithm. Defaults to 'commute'.

                - 'commute': use the commute hamiltonian to optimize the problem.

                - 'cyclic': use the cyclic hamiltonian to optimize the problem.

                - 'penalty': use the penalty hamiltonian to optimize the problem.

            penalty_lambda (float, optional): the penalty parameter for the algorithm. Defaults to None.
        """
        """
        Set the optimization method for the algorithm.

        Args:
            type (str, optional): the optimization method for the algorithm. Defaults to 'commute'.

                - 'commute': use the commute hamiltonian to optimize the problem.

                - 'cyclic': use the cyclic hamiltonian to optimize the problem.

                - 'penalty': use the penalty hamiltonian to optimize the problem.

            penalty_lambda (float, optional): the penalty parameter for the algorithm. Defaults to None.
        """
        self.algorithm_optimization_method = type
        self.penalty_lambda = penalty_lambda

    def add_binary_variables(self, name:str, shape:List[int]):
        """ Add `num` 0-1 variables. 
        
        If the variable is already in the dictionary, provide feedback and then skip this addition.

        Args:
            name (str): the label of varibles. 
            shape (int): the shape of the variables array. 

        Example:
            add_binary_variables('x', [5]): ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']
            add_binary_variables('x', [2, 2]): [['x_0_0', 'x_0_1'], ['x_1_0', 'x_1_1']]
        """
        if name in self.variable_name_set:
            iprint(f'varible {name} already exist')
            return None
        self.variable_name_set.add(name)
        def gnrt_variables(prefix, shape):
            if len(shape) == 1:
                var_name_list = [f"{prefix}_{i}" for i in range(shape[0])]
                var_list = [Variable(var) for var in var_name_list]
                for var_name in var_name_list:
                    self.variables_idx[var_name] = self.current_var_idx
                    self.current_var_idx += 1
                self.variables.extend(var_list)
                return var_list
            else:
                variables = []
                for i in range(shape[0]):
                    sub_indices = gnrt_variables(prefix=f"{prefix}_{i}", shape=shape[1:])
                    variables.append(sub_indices)
                return variables
        BVars = gnrt_variables(name, shape)
        return BVars
    
    def add_binary_variable(self, name:str):
        """ Add one 0-1 variable

        Args:
            name (str): the name to label variable
        """
        var = Variable(name)
        if name in self.variables_idx:
            iprint(f'varible {name} already exist')
            return None
        self.variables.append(var)
        self.variables_idx[name] = self.current_var_idx
        self.current_var_idx += 1
        return var
    # 输入变量对象, 输出变量在问题对象变量列表中的下标, 算cost等用
    def var_to_idex(self, var: Variable):
        return self.variables_idx[var.name]
    
    def add_objective(self, expr_or_func):
        # 待拓展非线性目标函数表达式解析
        if isinstance(expr_or_func, str):
            coefficients = np.zeros(len(self.variables) + 1)
            for sign, term in split_expr(expr_or_func.split('==')[0]):
                sign = 1 if sign == '+' else -1
                if '*' in term:
                    coefficient, variable = term.split('*')
                    coefficients[self.variables_idx[variable]] = sign * int(coefficient)
                else:
                    variable = term
                    coefficients[self.variables_idx[variable]] = sign
            coefficients[-1] = int(expr_or_func.split('==')[1])
            (coefficients)
            self.add_linear_objective(coefficients)
        elif callable(expr_or_func):
            self.objective_func = expr_or_func
        else:
            raise ValueError("Unsupported objective_func type. Expected either a string or a callable function.")

    def add_linear_objective(self, coefficients: Iterable):
        """add the objective function to the problem
        Args:
            coefficients (Iterable): [c_0, c_1,..., c_n-1] represents c_0*x_0 + c_1*x_1 + ... + c_n-1*x_n-1
        """
        for i in range(len(coefficients)):
            self.objective_func_term_list[0].append(([i], coefficients[i]))
            # self.objective_matrix[i][i] = coefficients
    
    def add_nonlinear_objective(self, term_list: Iterable, coefficient):
        self.objective_func_term_list[len(term_list) - 1].append((term_list, coefficient))

    def add_eq_constraint(self, coefficients: Iterable, variable):
        """_summary_

        Args:
            coefficients (Iterable): _description_
        """
        pass

    def add_constraint(self, expr):
        # 待拓展非线性约束表达式解析
        """
        Add a constraint to the optimization problem.

        Args:
            expr : A string representing a linear expression with variables and coefficients, followed by an equality or inequality sign.
                   e.g., '2*x_0 + 3*x_1 + 3*x_2 == 1'.
        """
        ## extract the coefficients and the variables from the expression like 2*x_0 + 3*x_1 + 3*x_2 == 1
        iprint(expr)
        coefficients = np.zeros(len(self.variables) + 1)
        ## parse the expression:2*x_0 + 3*x_1 + 3*x_2 == 1
        for sign,term in split_expr(expr.split('==')[0]):
            sign = 1 if sign == '+' else -1
            if '*' in term:
                coefficient,variable = term.split('*')
                coefficients[self.variables_idx[variable]] = sign*int(coefficient)
            else:
                variable = term
                coefficients[self.variables_idx[variable]] = sign
        coefficients[-1] = int(expr.split('==')[1])
        iprint(coefficients)
        self.add_linear_constraints(coefficients)
        ## get the coefficients and the variables

    def add_linear_constraint(self, coefficients: Iterable):
        self.linear_constraints.append(coefficients)

    @property
    def linear_constraints(self):
        # 子类自建linear_constraint 再分类到 for_cyclic & for_others
        if self._linear_constraints is None:
            return []
        return self._linear_constraints

    
    @property
    def constraints_classify_cyclic_others(self):
        if self._constraints_classify_cyclic_others is None:
            if self.opt_mtd == 'standard':
                linear_constraints = self.linear_constraints
            elif self.opt_mtd == 'dichotomy':
                linear_constraints = self.dctm_linear_constraints
            self._constraints_classify_cyclic_others = [[] for _ in range(2)]
            seen_indices = set()
            for cstrt in linear_constraints:
                if self.opt_mtd == 'standard':
                    assert len(cstrt) == 1 + len(self.variables)
                # 二分法减少一个比特
                elif self.opt_mtd == 'dichotomy':
                    assert len(cstrt) == 1 + len(self.variables) - self.num_frozen_qubit
                
                non_zero_indices = np.nonzero(cstrt[:-1])[0]
                if (set(cstrt[:-1]).issubset({0, 1}) or set(cstrt[:-1]).issubset({0, -1})) and not any(index in seen_indices for index in non_zero_indices):
                    assert cstrt[-1] >= 0
                    seen_indices.update(non_zero_indices)
                    self._constraints_classify_cyclic_others[0].append(cstrt)
                else:
                    self._constraints_classify_cyclic_others[1].append(cstrt)
        return self._constraints_classify_cyclic_others
        

    @property
    def driver_bitstr(self):
        if self._driver_bitstr is None:
            if self.fastsolve:
                self._driver_bitstr = self.fast_solve_driver_bitstr()
            # 如果不使用解析的解系, 高斯消元求解
            else:
                self._driver_bitstr = find_basic_solution(self.linear_constraints[:,:-1]) if len(self.linear_constraints) > 0 else []
        return self._driver_bitstr
    @property
    def dctm_driver_bitstr(self):
        return find_basic_solution(self.dctm_linear_constraints[:,:-1]) if len(self.dctm_linear_constraints) > 0 else []

    def get_feasible_solution(self):
        ## find a feasible solution for the linear_constraints
        for i in range(1 << len(self.variables)):
            bitstr = [int(j) for j in list(bin(i)[2:].zfill(len(self.variables)))]
            if all([np.dot(bitstr,constr[:-1]) == constr[-1] for constr in self.linear_constraints]):
                return bitstr
        return
    
    def get_best_cost(self):
        state_0 = [int(j) for j in list(bin(0)[2:].zfill(len(self.variables)))]
        best_cost = self.objective_penalty(state_0)
        best_solution_list = [state_0]
        for i in range(1, 1 << len(self.variables)):
            bitstr = [int(j) for j in list(bin(i)[2:].zfill(len(self.variables)))]
            record_value = self.objective_penalty(bitstr)
            if record_value < best_cost:
                best_cost = record_value
                best_solution_list = [bitstr]
            elif record_value == best_cost:
                best_solution_list.append(bitstr)
        iprint(f'best_cost: {best_cost}')
        iprint(f'best_solution: {best_solution_list}')
        return best_cost

    def optimize(self, optimizer_option: OptimizerOption, circuit_option: CircuitOption) -> None: 
        circuit_option.num_qubits = len(self.variables)
        circuit_option.algorithm_optimization_method = self.algorithm_optimization_method
        circuit_option.penalty_lambda = self.penalty_lambda
        circuit_option.feasiable_state = self.get_feasible_solution()
        circuit_option.objective_func_term_list = self.objective_func_term_list
        circuit_option.linear_constraints = self.linear_constraints
        circuit_option.constraints_for_cyclic = self.constraints_classify_cyclic_others[0]
        circuit_option.constraints_for_others = self.constraints_classify_cyclic_others[1]
        circuit_option.Hd_bits_list  = self.driver_bitstr

        iprint(f'fsb_state: {circuit_option.feasiable_state}') #-
        iprint(f'driver_bit_stirng:\n {self.driver_bitstr}') #-
        objective_func_map = {
            'penalty': self.objective_penalty,
            'cyclic': self.objective_cyclic,
            'commute': self.objective_commute,
            'HEA': self.objective_penalty
        }
        if self.algorithm_optimization_method in objective_func_map:
            circuit_option.objective_func = objective_func_map.get(self.algorithm_optimization_method)

        try:
            collapse_state, probs, iteration_count = solve(optimizer_option, circuit_option)
        except QuickFeedbackException as qfe:
            return qfe.data
        self.collapse_state=collapse_state
        self.probs=probs
        # collapse_state_str = [''.join([str(x) for x in state]) for state in collapse_state]
        # iprint(dict(zip(collapse_state_str, probs)))


        # 找到最优解和最优解的cost / by groubi
        best_cost = self.get_best_cost()
        iprint(f'best_cost: {best_cost}')
        mean_cost = 0
        best_solution_probs = 0
        for c, p in zip(collapse_state, probs):
            pcost = self.cost_dir * self.objective_penalty(c)
            if p >= 1e-3:
                iprint(f'{c}: {pcost} - {p}')
            if pcost == best_cost:
                best_solution_probs += p
            mean_cost += pcost * p
        best_solution_probs *= 100

        #+ 输出最大概率解
        maxprobidex = np.argmax(probs)
        max_prob_solution = collapse_state[maxprobidex]
        cost = self.cost_dir * circuit_option.objective_func(max_prob_solution)
        iprint(f"max_prob_solution: {max_prob_solution}, cost: {cost}, max_prob: {probs[maxprobidex]:.2%}") #-
        iprint(f'best_solution_probs: {best_solution_probs}')
        iprint(f"mean_cost: {mean_cost}")
        in_constraints_probs = 0
        for cs, pr in zip(self.collapse_state, self.probs):
            try:
                if all([np.dot(cs,constr[:-1]) == constr[-1] for constr in self.linear_constraints]):
                    in_constraints_probs += pr
            except:
                print(self.linear_constraints, cs)
        in_constraints_probs *= 100
        iprint(f'in_constraint_probs: {in_constraints_probs}')
        ARG = abs((mean_cost - best_cost) / best_cost)
        iprint(f'ARG: {ARG}')
        return ARG, in_constraints_probs, best_solution_probs, iteration_count

    def dichotomy_optimize(self, optimizer_option: OptimizerOption, circuit_option: CircuitOption, num_frozen_qubit: int = 1) -> None: 
        self.opt_mtd = 'dichotomy'
        self.num_frozen_qubit = num_frozen_qubit
        iprint(self.driver_bitstr)
        iprint()
        # 最多非零元素的列索引, 对该比特冻结 | 注意, 不是约束最多的列，是driver_bitstr最多的列
        iprint(self.driver_bitstr)
        non_zero_counts = np.count_nonzero(self.driver_bitstr, axis=0)
        sorted_indices = np.argsort(non_zero_counts)[::-1][:num_frozen_qubit]
        self.frozen_idx_list = sorted(sorted_indices)
        # self.frozen_idx_list = [1, 4]

        def find_feasible_solution_with_gurobi(A, fixed_values=None):
            num_vars = A.shape[1] - 1  # Number of variables
            # Create a new model
            model = gp.Model("feasible_solution")
            model.setParam('OutputFlag', 0)
            # Create variables
            variables = []
            for i in range(num_vars):
                variables.append(model.addVar(vtype=gp.GRB.BINARY, name=f"x{i}"))
            # Set objective (minimization problem, but no objective here since we just need a feasible solution)
            # Add constraints
            for i in range(A.shape[0]):
                lhs = gp.quicksum(A[i, j] * variables[j] for j in range(num_vars))
                rhs = A[i, -1]
                model.addConstr(lhs == rhs)
                
            # Add fixed values constraints
            if fixed_values:
                idx, fix = fixed_values
                for i, f in zip(idx, fix):
                    model.addConstr(variables[i] == f)
            
            model.optimize()
            
            if model.status == gp.GRB.Status.OPTIMAL:
                # Retrieve solution
                solution = [int(variables[i].x) for i in range(num_vars)]
                return solution
            else:
                return None
        ARG_list = []
        in_constraints_probs_list = []
        best_solution_probs_list = []
        iteration_count_list = []
        objective_func_map = {
            'penalty': self.objective_penalty,
            'cyclic': self.objective_cyclic,
            'commute': self.objective_commute,
            'HEA': self.objective_penalty
        }
        # 找到最优解的cost / by groubi
        best_cost = self.get_best_cost()
        for i in range(2**num_frozen_qubit):
            self.frozen_state_list = [int(j) for j in list(bin(i)[2:].zfill(num_frozen_qubit))]
            # 调整约束矩阵 1 修改常数列(c - frozen_state * frozen_idx), 2 剔除 frozen 列
            dctm_linear_constraints = self.linear_constraints.copy()
            iprint(f'self.linear_constraints:\n {self.linear_constraints}')
            for idx, state in zip(self.frozen_idx_list, self.frozen_state_list):
                dctm_linear_constraints[:,-1] -= dctm_linear_constraints[:, idx] * state
            self.dctm_linear_constraints = np.delete(dctm_linear_constraints, self.frozen_idx_list, axis=1)
            iprint(f'self.dichotomy_linear_constraints:\n {self.dctm_linear_constraints}')
            dctm_feasible_solution = find_feasible_solution_with_gurobi(self.dctm_linear_constraints)
            if dctm_feasible_solution is None:
                continue
            # circuit_option.feasiable_state = np.delete(self.get_feasible_solution(), self.frozen_idx, axis=0)
            circuit_option.feasiable_state = dctm_feasible_solution

            circuit_option.num_qubits = len(self.variables) - len(self.frozen_idx_list)
            circuit_option.algorithm_optimization_method = self.algorithm_optimization_method
            circuit_option.penalty_lambda = self.penalty_lambda
            #+++ 这样只冻结了一种形态, 另一种形态待补
            iprint(f"frzoen_idx_list{self.frozen_idx_list}")
            iprint(f"frzoen_state_list{self.frozen_state_list}")
            iprint("feasible:", circuit_option.feasiable_state)
            # 处理剔除 frozen_qubit 后的目标函数    
            def process_objective_term_list(objective_iterm_list, frozen_idx_list, frozen_state_list):
                process_list = []
                zero_indices = [frozen_idx_list[i] for i, x in enumerate(frozen_state_list) if x == 0]
                nonzero_indices = [i for i in frozen_idx_list if i not in zero_indices]
                frozen_idx_list = np.array(frozen_idx_list)
                for dimension in objective_iterm_list:
                    dimension_list = []
                    for objective_term in dimension:
                        if any(idx in objective_term[0] for idx in zero_indices):
                            # 如果 frozen_state == 0 且 x 在内层列表中，移除整个iterm
                            continue
                        else:
                            # 如果 frozen_state == 1 且 x 在内层列表中，移除iterm中的 x
                            iterm = [varbs for varbs in objective_term[0] if varbs not in nonzero_indices]
                            iterm = [x - np.sum(frozen_idx_list < x) for x in iterm]
                            if iterm:
                                dimension_list.append((iterm, objective_term[1]))
                    # 空维度也要占位
                    process_list.append(dimension_list)
                return process_list
            circuit_option.objective_func_term_list = process_objective_term_list(self.objective_func_term_list, self.frozen_idx_list, self.frozen_state_list)
            iprint('term_list', circuit_option.objective_func_term_list)
            circuit_option.constraints_for_cyclic = self.constraints_classify_cyclic_others[0]
            circuit_option.constraints_for_others = self.constraints_classify_cyclic_others[1]
            circuit_option.Hd_bits_list = to_row_echelon_form(self.dctm_driver_bitstr)
            iprint(f'dctm_driver_bitstr:\n{self.dctm_driver_bitstr}') #-
            iprint(f'Hd_bits_list:\n{circuit_option.Hd_bits_list}') #-

            def dctm_objective_func_map(method: str):
                def dctm_objective_func(variables: Iterable):
                    def insert_states(filtered_list, idx_list, state_list):
                        result = []
                        state_index = 0
                        filtered_index = 0

                        for i in range(len(filtered_list) + len(state_list)):
                            if i in idx_list:
                                result.append(state_list[state_index])
                                state_index += 1
                            else:
                                result.append(filtered_list[filtered_index])
                                filtered_index += 1
                        return result
                    return objective_func_map.get(method)(insert_states(variables, self.frozen_idx_list, self.frozen_state_list))
                return dctm_objective_func
            circuit_option.objective_func = dctm_objective_func_map(self.algorithm_optimization_method)
            
            if len(circuit_option.Hd_bits_list) == 0:
                cost = self.cost_dir * circuit_option.objective_func(circuit_option.feasiable_state)
                ARG = abs((cost - best_cost) / best_cost)
                best_solution_probs = 100 if cost == best_cost else 0
                in_constraints_probs = 100 if all([np.dot(circuit_option.feasiable_state, constr[:-1]) == constr[-1] for constr in self.dctm_linear_constraints]) else 0
                ARG_list.append(ARG)
                best_solution_probs_list.append(best_solution_probs)
                in_constraints_probs_list.append(100)
                iteration_count_list.append(0)
                continue
            ###################################

                    
            try:
                collapse_state, probs, iteration_count = solve(optimizer_option, circuit_option)
            except QuickFeedbackException as qfe:
                return qfe.data
            self.collapse_state=collapse_state
            self.probs=probs
            # 最优解的cost / by groubi
            iprint(f'best_cost: {best_cost}')
            mean_cost = 0
            best_solution_probs = 0
            for c, p in zip(collapse_state, probs):
                pcost = self.cost_dir * dctm_objective_func_map('penalty')(c)
                if p >= 1e-3:
                    iprint(f'{c}: {pcost} - {p}')
                if pcost == best_cost:
                    best_solution_probs += p
                mean_cost += pcost * p
            best_solution_probs *= 100

            #+ 输出最大概率解
            maxprobidex = np.argmax(probs)
            max_prob_solution = collapse_state[maxprobidex]
            cost = self.cost_dir * circuit_option.objective_func(max_prob_solution)
            iprint(f"max_prob_solution: {max_prob_solution}, cost: {cost}, max_prob: {probs[maxprobidex]:.2%}") #-
            iprint(f'best_solution_probs: {best_solution_probs}')
            iprint(f"mean_cost: {mean_cost}")
            in_constraints_probs = 0
            for cs, pr in zip(self.collapse_state, self.probs):
                if all([np.dot(cs, constr[:-1]) == constr[-1] for constr in self.dctm_linear_constraints]):
                    in_constraints_probs += pr
            in_constraints_probs *= 100
            iprint(f'in_constraint_probs: {in_constraints_probs}')
            ARG = abs((mean_cost - best_cost) / best_cost)
            iprint(f'ARG: {ARG}')
            ARG_list.append(ARG)
            in_constraints_probs_list.append(in_constraints_probs)
            best_solution_probs_list.append(best_solution_probs)
            iteration_count_list.append(iteration_count)
        return ARG_list, in_constraints_probs_list, best_solution_probs_list, iteration_count_list  




