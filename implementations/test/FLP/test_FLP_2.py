import csv
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.models import CircuitOption, OptimizerOption
optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
circuit_option = CircuitOption(
    num_layers=0,
    need_draw=False,
    use_decompose=True,
    circuit_type='qiskit',
    mcx_mode='linear',
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
) 
methods = ['penalty', 'cyclic', 'commute', 'HEA']
problems = [
    FLP(1, 2, [[10, 2]], [2, 2]),
    FLP(2, 2, [[10, 2], [1, 20]], [1, 1]),
    FLP(2, 3, [[10, 2, 3], [1, 20, 3]], [1, 1, 1]),
    # FLP(3, 3, [[10, 2, 3], [10, 2, 3], [10, 2, 3]], [1, 1, 1]),
    # FLP(3, 4, [[10, 2, 3, 1], [10, 2, 3, 1], [10, 2, 3, 1], [10, 2, 3, 1]], [1, 1, 1, 1]),
    # FLP(5, 5, [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], [1, 1, 1, 1, 1])
]
csv_data = []
headers = ['Problem_Size', 'Layers', 'Variables', 'Constraints']
evaluation_metrics = ['ARG', 'in_constraints_probs', 'best_solution_probs']

for metric in evaluation_metrics:
    for method in methods:
        headers.append(f'{method}_{metric}')

csv_data.append(headers)
num_layers_range = range(1, 6)
for num_layers in num_layers_range:
    for flp in problems:
        data = [[] for _ in range(len(methods))]
        
        print(f'问题规模:{flp.m} * {flp.n}')
        print(f'Layers: {num_layers}')
        print(f'v: {flp.num_variables}, c: {len(flp.linear_constraints)}')

        row = [f'{flp.m} * {flp.n}', num_layers, flp.num_variables, len(flp.linear_constraints)]
        circuit_option.num_layers = num_layers
        metric_data = {metric: [] for metric in evaluation_metrics}

        for method in methods:
            flp.set_algorithm_optimization_method(method, 400)
            ARG, in_constraints_probs, best_solution_probs = flp.optimize(optimizer_option, circuit_option)
            metric_data['ARG'].append(ARG)
            metric_data['in_constraints_probs'].append(in_constraints_probs)
            metric_data['best_solution_probs'].append(best_solution_probs)

        for metric in evaluation_metrics:
            row.extend(metric_data[metric])
        csv_data.append(row)

csv_filename = 'flp_results_evaluation_0.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'Data has been written to {csv_filename}')
