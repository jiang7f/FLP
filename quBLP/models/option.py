from typing import List, Callable
from dataclasses import dataclass, field

@dataclass
class OptimizerOption:
    params_optimization_method: str
    max_iter: int
    learning_rate: float
    beta1: float
    beta2: float
    cost_function: Callable = None
    num_params: int = None

@dataclass
class CircuitOption:
    circuit_type: str
    num_qubits: int
    num_layers: int
    objective: Callable
    algorithm_optimization_method: str
    optimization_direction: str
    use_decompose: bool
    need_draw: bool
    mcx_mode: str ## 'constant' for 2 additional ancillas with linear depth, 'linear' for n -1 additional ancilla with logarithmic depth
    debug: bool = False
    penalty_lambda: float = None
    use_Ho_gate_list: bool = True
    Ho_gate_list: List[List[int]] = field(default_factory=list)
    feasiable_state: List[int] = field(default_factory=list)
    linear_objective_vector: List[float] = field(default_factory=list)
    nonlinear_objective_matrix: List[List[float]] = field(default_factory=list)
    constraints: List[List[float]] = field(default_factory=list)
    constraints_for_cyclic: List[List[float]] = field(default_factory=list)
    constraints_for_others: List[List[float]] = field(default_factory=list)
    Hd_bits_list: List[List[int]] = field(default_factory=list)