import csv
from quBLP.problemtemplate import GraphColoringProblem as GCP
from quBLP.models import CircuitOption, OptimizerOption

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)

methods = ['penalty', 'cyclic', 'commute', 'HEA']
feedback = ['depth', 'culled_depth']

problems = [
    GCP(3, [[0, 1]]), 
    GCP(3, [[0, 1], [1, 2]]),
    GCP(4, [[0, 1]]),
    GCP(4, [[0, 1], [1, 2]]),
    GCP(4, [[0, 1], [1, 2], [2, 3]])
]

csv_data = []
headers = ['Problem_Size', 'Variables', 'Constraints', 'Layers', 'mcx_mode']
for dict_term in feedback:
    for method in methods:
        headers.extend([f'{method}_{dict_term}'])
csv_data.append(headers)

layers = range(1, 2)

mcx_modes = ['constant', 'linear']

for mcx_mode in mcx_modes:
    for gcp in problems:
        data = [[] for _ in range(len(methods))]
        for idx, method in enumerate(methods):
            gcp.set_algorithm_optimization_method(method, 400)
            for num_layers in layers:
                circuit_option = CircuitOption(
                    num_layers=num_layers,
                    need_draw=False,
                    use_decompose=True,
                    circuit_type='qiskit',
                    mcx_mode=mcx_mode,
                    backend='AerSimulator',
                    feedback=['depth', 'culled_depth', 'latency']
                )
                data[idx].append(gcp.optimize(optimizer_option, circuit_option))
        
        print(f'问题规模: {gcp.num_graphs}, {gcp.pairs_adjacent}')
        print(f'v: {gcp.num_variables}, c: {len(gcp.linear_constraints)}')

        for num_layers in layers:
            row = [f'{gcp.num_graphs}, {gcp.pairs_adjacent}', gcp.num_variables, len(gcp.linear_constraints), num_layers, mcx_mode]
            for dict_term in feedback:
                for idx, method in enumerate(methods):
                    row.extend([data[idx][l - 1][dict_term] for l in layers])
            csv_data.append(row)

csv_filename = '../data/GCP/gcp_results_depth.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'Data has been written to {csv_filename}')
