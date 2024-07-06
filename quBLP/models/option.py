from typing import List, Callable, Tuple
from dataclasses import dataclass, field

@dataclass
class OptimizerOption:
    params_optimization_method: str = 'COBYLA'
    max_iter: int = 30
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    # 
    circuit_cost_function: Callable = None
    num_params: int = None

@dataclass
class CircuitOption:
    num_layers: int = 2
    need_draw: bool = False
    use_decompose: bool = False
    circuit_type: str = 'qiskit'
    mcx_mode: str = 'constant'  # 'constant' for 2 additional ancillas with linear depth, 'linear' for n-1 additional ancillas with logarithmic depth
    backend: str = 'FakeAlmadenV2' #'FakeQuebec' # 'AerSimulator'\
    feedback: List = field(default_factory=list)
    # 
    num_qubits: int = 0
    algorithm_optimization_method: str = 'commute'
    penalty_lambda: float = None
    objective_func: Callable = None
    feasiable_state: List[int] = field(default_factory=list)
    objective_func_term_list: List[List[Tuple[List[int], float]]] = field(default_factory=list)
    linear_constraints: List[List[float]] = field(default_factory=list)
    constraints_for_cyclic: List[List[float]] = field(default_factory=list)
    constraints_for_others: List[List[float]] = field(default_factory=list)
    Hd_bits_list: List[List[int]] = field(default_factory=list)