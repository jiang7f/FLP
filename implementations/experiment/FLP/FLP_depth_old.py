import csv
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.models import CircuitOption, OptimizerOption

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)

methods = ['penalty', 'cyclic', 'commute', 'HEA']
feedback = ['depth', 'culled_depth']

problems = [
    FLP(1, 2, [[10, 2]],[2, 2]), 
    FLP(2, 2, [[10, 2],[1, 20]],[1, 1]),
    FLP(2, 3, [[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
    FLP(3, 3, [[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1]),
    FLP(3, 4, [[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1]],[1, 1, 1, 1]),
]

csv_data = []
headers = ['Problem_Size', 'Layers', 'Variables', 'Constraints', 'mcx_mode']
for dict_term in feedback:
    for method in methods:
        headers.extend([f'{method}_{dict_term}'])
csv_data.append(headers)

layers = range(1, 2)

mcx_modes = ['constant', 'linear']
feedback=['depth', 'culled_depth']
for mcx_mode in mcx_modes:
    for flp in problems:
        data = [[] for _ in range(len(methods))]
        for idx, method in enumerate(methods):
            flp.set_algorithm_optimization_method(method, 400)
            for num_layers in layers:
                circuit_option = CircuitOption(
                    num_layers=num_layers,
                    need_draw=False,
                    use_decompose=True,
                    circuit_type='qiskit',
                    mcx_mode=mcx_mode,
                    backend='AerSimulator',
                    feedback=feedback,
                )
                data[idx].append(flp.optimize(optimizer_option, circuit_option))
        
        print(f'问题规模: {flp.num_demands} * {flp.num_facilities}')
        print(f'v: {flp.num_variables}, c: {len(flp.linear_constraints)}')

        for num_layers in layers:
            row = [f'{flp.num_demands}, {flp.num_facilities}', num_layers, flp.num_variables, len(flp.linear_constraints), mcx_mode]
            for dict_term in feedback:
                for idx, method in enumerate(methods):
                    row.extend([data[idx][l - 1][dict_term] for l in layers])
            csv_data.append(row)

csv_filename = '../data/FLP/flp_results_depth.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'Data has been written to {csv_filename}')
