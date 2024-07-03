from typing import List, Callable, Tuple
from dataclasses import dataclass, field

@dataclass
class OptimizerOption:
    params_optimization_method: str
    max_iter: int
    learning_rate: float
    beta1: float
    beta2: float
    circuit_cost_function: Callable = None
    num_params: int = None

@dataclass
class CircuitOption:
    circuit_type: str
    num_qubits: int
    num_layers: int
    algorithm_optimization_method: str
    objective_func: Callable
    use_decompose: bool
    need_draw: bool
    mcx_mode: str # 'constant' for 2 additional ancillas with linear depth, 'linear' for n -1 additional ancilla with logarithmic depth
    debug: bool = False
    penalty_lambda: float = None
    feasiable_state: List[int] = field(default_factory=list)
    objective_func_term_list: List[List[Tuple[List[int], float]]] = field(default_factory=list)
    linear_constraints: List[List[float]] = field(default_factory=list)
    constraints_for_cyclic: List[List[float]] = field(default_factory=list)
    constraints_for_others: List[List[float]] = field(default_factory=list)
    Hd_bits_list: List[List[int]] = field(default_factory=list)
    backend: str = 'FakeAlmadenV2' #'FakeQuebec' # 'AerSimulator'