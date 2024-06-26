## This file defines a problem of binary constraint optimization 
import numpy as np
import itertools
from typing import Iterable, List, Callable,  Union
from ..utils.linear_system import find_basic_solution
from ..utils.parse_expr import split_expr
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
        if isinstance(other,Expression):
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
    def __init__(self,fastsolve=False):
        """ a constrainted binary optimization problem

        Args:
            fastsolve (bool, optional): whether to use fast solve method to get the bitstring for driver hamiltonian. Defaults to False.
        """
        self.fastsolve = fastsolve
        self.variables = []
        self.constraints_for_cyclic = [] # 用于存∑=x
        self.constraints_for_others = []
        # self.constraints yeld by @property
        self.linear_objective_vector = []
        self.nonlinear_objective_matrix = []
        self.objective = None
        self.variables_idx = {}
        self.current_var_idx = 0
        self.variable_name_set = set()
        self.algorithm_optimization_method = 'commute'
        self.penalty_lambda = 0
        self.collapse_state = None
        self.probs = None
        self.optimization_direction = None
        self.cost_dir = 0
        # self.Ho_gate_list yeld by @property
        self.objective_penalty = None
        self.objective_cyclic = None
        self.objective_commute = None
        pass

    def set_optimization_direction(self, dir):
        assert dir in ['min', 'max']
        self.optimization_direction = dir
        self.cost_dir = 1 if dir == 'min' else -1 if dir == 'max' else None

    def find_state_probability(self, state):
        index = self.collapse_state.index(state)
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
            print(f'varible {name} already exist')
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
            print(f'varible {name} already exist')
            return None
        self.variables.append(var)
        self.variables_idx[name] = self.current_var_idx
        self.current_var_idx += 1
        return var
    # 输入变量对象, 输出变量在问题对象变量列表中的下标, 算cost用
    def var_to_idex(self, var: Variable):
        return self.variables_idx[var.name]
    def add_constraint(self, expr):
        """
        Add a constraint to the optimization problem.

        Args:
            expr : A string representing a linear expression with variables and coefficients, followed by an equality or inequality sign.
                   e.g., '2*x_0 + 3*x_1 + 3*x_2 == 1'.
        """
        ## extract the coefficients and the variables from the expression like 2*x_0 + 3*x_1 + 3*x_2 == 1
        print(expr)
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
        coefficients[-1]= int(expr.split('==')[1])
        print(coefficients)
        self._add_linear_constraint(coefficients)
        ## get the coefficients and the variables

    def _add_linear_constraint(self, coefficients: Iterable):
        """add one linear constraint to the problem

        Args:
            coefficients (Iterable): [a_0, a_1,..., a_n-1, b] represents a_0*x_0 + a_1*x_1 + ... + a_n-1*x_n-1 == b
        """
        assert len(coefficients) == 1 + len(self.variables)
        if set(coefficients[:-1]).issubset({0, 1}):
            assert coefficients[-1] >= 0
            self.constraints_for_cyclic.append(coefficients)
        else:
            self.constraints_for_others.append(coefficients)
    @property
    def constraints(self):
        cstt = np.array(self.constraints_for_cyclic + self.constraints_for_others)
        return cstt
    
    def add_objective(self, expr):
        coefficients = np.zeros(len(self.variables) + 1)
        for sign,term in split_expr(expr.split('==')[0]):
            sign = 1 if sign == '+' else -1
            if '*' in term:
                coefficient,variable = term.split('*')
                coefficients[self.variables_idx[variable]] = sign*int(coefficient)
            else:
                variable = term
                coefficients[self.variables_idx[variable]] = sign
        coefficients[-1]= int(expr.split('==')[1])
        print(coefficients)
        self._add_linear_objective(coefficients)

    def _add_linear_objective(self, coefficients: Iterable):
        """add the objective function to the problem
        Args:
            coefficients (Iterable): [c_0, c_1,..., c_n-1] represents c_0*x_0 + c_1*x_1 + ... + c_n-1*x_n-1
        """
        self.linear_objective_vector = coefficients

    def add_eq_constraint(self, coefficients: Iterable, variable):
        """_summary_

        Args:
            coefficients (Iterable): _description_
        """
        pass

    @property
    def linear_constraints(self):
        return []
    @property
    def get_driver_bitstr(self):
        if self.fastsolve:
            return self.fast_solve_driver_bitstr()
        # 如果不使用解析的解系, 高斯消元求解
        basic_vector = find_basic_solution(self.constraints[:,:-1]) if len(self.constraints) > 0 else []
        return basic_vector
    def add_objective(self,objectivefunc):
        self.objective = objectivefunc
    def set_objective(self, expr):
        pass
    def get_feasible_solution(self):
        ## find a feasible solution for the constraints
        for i in range(1 << len(self.variables)):
            bitstr = [int(j) for j in list(bin(i)[2:].zfill(len(self.variables)))]
            if all([np.dot(bitstr,constr[:-1]) == constr[-1] for constr in self.constraints]):
                return bitstr
        return
    def optimize(self, params_optimization_method='Adam', max_iter=30, learning_rate=0.1, num_layers=2, need_draw=False, beta1=0.9, beta2=0.999, use_Ho_gate_list=False, use_decompose=False, circuit_type='pennylane',mcx_mode='constant', debug=True, backend='FakeAlmadenV2', pass_manager='topo') -> None: 

        self.feasiable_state = self.get_feasible_solution()
        print(f'fsb_state:{self.feasiable_state}') #-
        
        optimizer_option = OptimizerOption(
            params_optimization_method=params_optimization_method,
            max_iter=max_iter,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
        )

        circuit_option = CircuitOption(
            circuit_type=circuit_type,
            num_qubits=len(self.variables),
            num_layers=num_layers,
            objective=None,
            mcx_mode=mcx_mode,
            algorithm_optimization_method=self.algorithm_optimization_method,
            feasiable_state=self.feasiable_state,
            optimization_direction=self.optimization_direction,
            use_decompose=use_decompose,
            linear_objective_vector=self.linear_objective_vector,
            nonlinear_objective_matrix=self.nonlinear_objective_matrix,
            need_draw=need_draw,
            use_Ho_gate_list=use_Ho_gate_list,
            Ho_gate_list = self.Ho_gate_list,
            penalty_lambda = self.penalty_lambda,
            constraints = self.constraints,
            constraints_for_cyclic = self.constraints_for_cyclic,
            constraints_for_others = self.constraints_for_others,
            Hd_bits_list = self.get_driver_bitstr,
            debug=debug,
            backend=backend,
            pass_manager=pass_manager
        )

        objective_map = {
            'penalty': self.objective_penalty,
            'cyclic': self.objective_cyclic,
            'commute': self.objective_commute,
            'HEA': self.objective_penalty
        }
        circuit_option.objective = objective_map.get(self.algorithm_optimization_method)

        collapse_state, probs = solve(optimizer_option, circuit_option)
        #+ 输出最大概率解
        maxprobidex = np.argmax(probs)
        max_prob_solution = collapse_state[maxprobidex]
        cost = circuit_option.objective(max_prob_solution)
        print(collapse_state)
        print(f"max_prob_solution: {max_prob_solution}, cost: {cost}, max_prob: {probs[maxprobidex]:.2%}") #-
        self.collapse_state=collapse_state
        self.probs=probs

