import csv
from quBLP.problemtemplate import KPartitionProblem as KPP
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

problems = [
    KPP(3, [2, 1], [[0, 1], [1, 2]]),
    KPP(4, [3, 1], [[0, 1], [1, 2], [2, 3]]),
    KPP(5, [4, 1], [[0, 1], [1, 2], [2, 3], [3, 4]]),
    KPP(5, [3, 2], [[0, 1], [1, 2], [2, 3], [3, 4]]),
    KPP(5, [3, 1, 1], [[0, 1], [1, 2], [2, 3], [3, 4]])
]

csv_data = []

headers = ['Problem_Size', 'Variables', 'Constraints', 'Layers']
for dict_term in feedback:
    for method in methods:
        headers.extend([f'{method}_{dict_term}'])
csv_data.append(headers)

layers = range(1, 2)
for kpp in problems:
    data = [[] for _ in range(len(methods))]
    for idx, method in enumerate(methods):
        kpp.set_algorithm_optimization_method(method, 400)
        for num_layers in layers:
            circuit_option.num_layers = num_layers
            data[idx].append(kpp.optimize(optimizer_option, circuit_option))
    
    print(f'问题规模:{kpp.num_points}, {kpp.block_allot}, {kpp.pairs_connected}')
    print(f'v: {kpp.num_variables}, c: {len(kpp.linear_constraints)}')

    for num_layers in layers:
        row = [f'{kpp.num_points}, {kpp.block_allot}, {kpp.pairs_connected}', kpp.num_variables, len(kpp.linear_constraints), num_layers]
        for dict_term in feedback:
            for idx, method in enumerate(methods):
                row.extend([data[idx][l - 1][dict_term] for l in layers])
        csv_data.append(row)

csv_filename = 'kpp_results_depth.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'Data has been written to {csv_filename}')
