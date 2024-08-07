import csv
from quBLP.problemtemplate import One_Hdi as OH
from quBLP.models import CircuitOption, OptimizerOption
feedback=['depth', 'culled_depth']
optimizer_option = OptimizerOption(
    params_optimization_method='COBYLA',
    max_iter=150
)

circuit_option = CircuitOption(
    num_layers=1,
    need_draw=False,
    use_decompose=True,
    circuit_type='qiskit',
    mcx_mode='linear',
    backend='AerSimulator',  # 'FakeQuebec' # 'AerSimulator'
    feedback=feedback,
)
# methods = ['penalty', 'cyclic', 'commute', 'HEA']
problems = [OH(40, i) for i in range(2, 21, 2)]

csv_data = []
headers = ['num_qubits', 'num_shared_variables', 'Layers', 'num_constraint', 'depth', 'culled_depth']
csv_data.append(headers)
layers = range(1, 2)
for sv in problems:
    sv.set_algorithm_optimization_method('commute', 400)
    data = []
    for num_layers in layers:
        circuit_option.num_layers = num_layers
        data.append(sv.optimize(optimizer_option, circuit_option))
    
    print(f'问题规模:{sv.num_qubits}, {sv.mcx_constraint}, 约束数量: {len(sv.linear_constraints)}')

    for num_layers in layers:
        row = [sv.num_qubits, sv.mcx_constraint, num_layers, len(sv.linear_constraints)]
        for dict_term in feedback:
            row.extend([data[l - 1][dict_term] for l in layers])
        csv_data.append(row)

csv_filename = '../../data/others/shared_linear_depth.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)