import csv
from quBLP.problemtemplate import KPartitionProblem as KPP
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
    # feedback=['depth', 'culled_depth', 'latency'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
feedback = ['depth', 'culled_depth']

problems = [
    KPP(3,[2, 1],[[0, 1], [1,2]]), 
    KPP(4,[3, 1],[[0, 1], [1, 2],[2, 3]]),
    KPP(5,[4, 1],[[0, 1], [1, 2], [2, 3], [3, 4]]),
    KPP(5 ,[3, 2], [[0, 1], [1, 2], [2, 3], [3, 4]]),
    KPP(6,[3, 1, 1], [[0, 1], [1, 2], [2, 3], [3, 4]])
]

csv_data = []
headers = ['Problem_Size', 'Layers', 'Variables', 'Constraints']
evaluation_metrics = ['ARG', 'in_constraints_probs', 'best_solution_probs']

for metric in evaluation_metrics:
    for method in methods:
        headers.append(f'{method}_{metric}')

csv_data.append(headers)
num_layers_range = range(1, 3)
for num_layers in num_layers_range:
    for kpp in problems:
        data = [[] for _ in range(len(methods))]
        
        print(f'问题规模:{kpp.num_points}, {kpp.block_allot}, {kpp.pairs_connected}')
        print(f'Layers: {num_layers}')
        print(f'v: {kpp.num_variables}, c: {len(kpp.linear_constraints)}')

        row = [f'{kpp.num_points}, {kpp.block_allot}, {kpp.pairs_connected}', num_layers, kpp.num_variables, len(kpp.linear_constraints)]
        circuit_option.num_layers = num_layers
        metric_data = {metric: [] for metric in evaluation_metrics}

        for method in methods:
            kpp.set_algorithm_optimization_method(method, 400)
            ARG, in_constraints_probs, best_solution_probs = kpp.optimize(optimizer_option, circuit_option)
            metric_data['ARG'].append(ARG)
            metric_data['in_constraints_probs'].append(in_constraints_probs)
            metric_data['best_solution_probs'].append(best_solution_probs)

        for metric in evaluation_metrics:
            row.extend(metric_data[metric])
        csv_data.append(row)

csv_filename = 'kpp_results_evaluation.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'Data has been written to {csv_filename}')