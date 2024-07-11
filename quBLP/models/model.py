## This file defines a problem of binary constraint optimization 
from quBLP.utils import iprint
import numpy as np
import itertools
from typing import Iterable, List, Callable,  Union
from ..utils.linear_system import find_basic_solution
from ..utils.parse_expr import split_expr
from ..utils import QuickFeedbackException
from ..solvers import solve
from dataclasses import dataclass, field
from .option import OptimizerOption, CircuitOption

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
        self.fastsolve = fastsolve
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
        pass

    def set_optimization_direction(self, dir):
        assert dir in ['min', 'max']
        self.cost_dir = 1 if dir == 'min' else -1 if dir == 'max' else None

    def find_state_probability(self, state):
        try:
            index = self.collapse_state.index(state)
        except ValueError:
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
            self.objective_func_term_list[0].append([[i], coefficients[i]])
            # self.objective_matrix[i][i] = coefficients
    
    def add_nonlinear_objective(self, term_list: Iterable, coefficient):
        self.objective_func_term_list[len(term_list) - 1].append([term_list, coefficient])

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
            self._constraints_classify_cyclic_others = [[] for _ in range(2)]
            for cstrt in self.linear_constraints:
                assert len(cstrt) == 1 + len(self.variables)
                if set(cstrt[:-1]).issubset({0, 1}):
                    assert cstrt[-1] >= 0
                    self._constraints_classify_cyclic_others[0].append(cstrt)
                else:
                    self._constraints_classify_cyclic_others[1].append(cstrt)
        return self._constraints_classify_cyclic_others
        

    @property
    def get_driver_bitstr(self):
        if self.fastsolve:
            return self.fast_solve_driver_bitstr()
        # 如果不使用解析的解系, 高斯消元求解
        basic_vector = find_basic_solution(self.linear_constraints[:,:-1]) if len(self.linear_constraints) > 0 else []
        return basic_vector

    def get_feasible_solution(self):
        ## find a feasible solution for the linear_constraints
        for i in range(1 << len(self.variables)):
            bitstr = [int(j) for j in list(bin(i)[2:].zfill(len(self.variables)))]
            if all([np.dot(bitstr,constr[:-1]) == constr[-1] for constr in self.linear_constraints]):
                return bitstr
        return

    def optimize(self, optimizer_option: OptimizerOption, circuit_option: CircuitOption) -> None: 
        circuit_option.num_qubits = len(self.variables)
        circuit_option.algorithm_optimization_method = self.algorithm_optimization_method
        circuit_option.penalty_lambda = self.penalty_lambda
        circuit_option.feasiable_state = self.get_feasible_solution()
        circuit_option.objective_func_term_list = self.objective_func_term_list
        circuit_option.linear_constraints = self.linear_constraints
        circuit_option.constraints_for_cyclic = self.constraints_classify_cyclic_others[0]
        circuit_option.constraints_for_others = self.constraints_classify_cyclic_others[1]
        circuit_option.Hd_bits_list  = self.get_driver_bitstr
        np.set_printoptions(threshold=np.inf, suppress=True, precision=4,  linewidth=300)
        iprint(f'fsb_state: {circuit_option.feasiable_state}') #-
        iprint(f'driver_bit_stirng:\n {self.get_driver_bitstr}') #-
        objective_func_map = {
            'penalty': self.objective_penalty,
            'cyclic': self.objective_cyclic,
            'commute': self.objective_commute,
            'HEA': self.objective_penalty
        }
        if self.algorithm_optimization_method in objective_func_map:
            circuit_option.objective_func = objective_func_map.get(self.algorithm_optimization_method)

        try:
            collapse_state, probs = solve(optimizer_option, circuit_option)
        except QuickFeedbackException as qfe:
            return qfe.data
        self.collapse_state=collapse_state
        self.probs=probs
        #+ 输出最大概率解
        maxprobidex = np.argmax(probs)
        max_prob_solution = collapse_state[maxprobidex]
        cost = circuit_option.objective_func(max_prob_solution)
        # collapse_state_str = [''.join([str(x) for x in state]) for state in collapse_state]
        # iprint(dict(zip(collapse_state_str, probs)))

        # 算最优解和最优解的cost
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
        mean_cost = 0
        for c, p in zip(collapse_state, probs):
            if p >= 1e-3:
                iprint(f'{c}: {self.objective_penalty(c)} - {p}')
            mean_cost += self.objective_penalty(c) * p
        iprint(f"max_prob_solution: {max_prob_solution}, cost: {cost}, max_prob: {probs[maxprobidex]:.2%}") #-
        best_solution_probs = sum([self.find_state_probability(best_solution) for best_solution in best_solution_list])
        best_solution_probs *= 100
        iprint(f'best_solution_probs: {best_solution_probs}')
        iprint(f"mean_cost: {mean_cost}")
        in_constraints_probs = 0
        for cs, pr in zip(self.collapse_state, self.probs):
            if all([np.dot(cs,constr[:-1]) == constr[-1] for constr in self.linear_constraints]):
                in_constraints_probs += pr
        in_constraints_probs *= 100
        iprint(f'in_constraint_probs: {in_constraints_probs}')
        ARG = abs((mean_cost - best_cost) / best_cost)
        iprint(f'ARG: {ARG}')
        return ARG, in_constraints_probs, best_solution_probs

        

