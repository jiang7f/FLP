import csv
from quBLP.problemtemplate import FacilityLocationProblem as FLP
from quBLP.models import CircuitOption, OptimizerOption

optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)
circuit_option = CircuitOption(
    num_layers=4,
    need_draw=False,
    use_decompose=True,
    circuit_type='qiskit',
    mcx_mode='linear',
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    feedback=['depth', 'culled_depth', 'latency'],
)
methods = ['penalty', 'cyclic', 'commute', 'HEA']
feedback = ['depth', 'culled_depth']

problems = [FLP(2,1,[[10, 2]],[2, 2]), 
            FLP(2,2,[[10, 2],[1, 20]],[1, 1]),
            FLP(3,2,[[10, 2, 3],[1, 20, 3]],[1, 1, 1]),
            FLP(3,3,[[10, 2, 3],[10, 2, 3],[10, 2, 3]],[1, 1, 1]),
            FLP(4,3,[[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1],[10, 2, 3, 1]],[1, 1, 1, 1]),]
            # FLP(5,5,[[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]],[1, 1, 1, 1, 1])]

csv_data = []

headers = ['Problem_Size', 'Variables', 'Constraints']
for dict_term in feedback:
    for method in methods:
        headers.extend([f'{method}_{dict_term}_Layer_{l}' for l in range(1, 2)])
csv_data.append(headers)

for flp in problems:
    data = [[] for _ in range(len(methods))]
    layers = range(1, 2)
    for idx, method in enumerate(methods):
        flp.set_algorithm_optimization_method(method, 400)
        for num_layers in layers:
            circuit_option.num_layers = num_layers
            data[idx].append(flp.optimize(optimizer_option, circuit_option))
    
    print(f'问题规模:{flp.m} * {flp.n}')
    print(f'v: {flp.num_variables}, c: {len(flp.linear_constraints)}')

    row = [f'{flp.m}, {flp.n}', flp.num_variables, len(flp.linear_constraints)]
    for dict_term in feedback:
        for idx, method in enumerate(methods):
            row.extend([data[idx][l - 1][dict_term] for l in layers])
    csv_data.append(row)

csv_filename = 'flp_results_depth.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'finished to {csv_filename}')
