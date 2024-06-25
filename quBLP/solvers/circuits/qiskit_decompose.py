from time import perf_counter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import transpile
from scipy.linalg import expm
import numpy as np


def apply_convert(qc, bit_string):
    num_qubits = len(bit_string)
    bit_string = bit_string[::-1]
    for i in range(0, num_qubits - 1):
        qc.cx(i + 1, i)
        if bit_string[i] == bit_string[i + 1]:
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.x(num_qubits - 1)

def apply_reverse(qc, bit_string):
    num_qubits = len(bit_string)
    bit_string = bit_string[::-1]
    qc.x(num_qubits - 1)
    qc.h(num_qubits - 1)
    for i in range(num_qubits - 2, -1, -1):
        if bit_string[i] == bit_string[i + 1]:
            qc.x(i)
        qc.cx(i + 1, i)

def decompose_phase_gate(circuit:QuantumCircuit, qr:QuantumRegister, ancilla:QuantumRegister, phase:float) -> QuantumCircuit:
    """
    Decompose a phase gate into a series of controlled-phase gates.
    Args:
        num_phase_qubits (int): the number of qubits that the phase gate acts on.
        phase (float): the phase angle of the phase gate.
        max_num_qubit_control (int): the maximum number of qubits that the controlled-phase gates can control.
    Returns:
        QuantumCircuit: the circuit that implements the decomposed phase gate.
    """
    num_phase_qubits = len(qr)
    if num_phase_qubits == 1:
        circuit.p(phase, 0)
    elif num_phase_qubits == 2:
        circuit.cp(phase, 0, 1)
    else:
        # convert into the multi-cx gate 
        # partition qubits into two sets
        half_num_qubit = num_phase_qubits // 2
        qr1 = qr[:half_num_qubit]
        qr2 = qr[half_num_qubit:]
        circuit.rz(-phase/2, ancilla[0])
        circuit.mcx(qr1, ancilla[0], ancilla[1], mode='recursion')
        circuit.rz(phase/2, ancilla[0])
        circuit.mcx(qr2, ancilla[0], ancilla[1], mode='recursion')
        circuit.rz(-phase/2, ancilla[0])
        circuit.mcx(qr1, ancilla[0], ancilla[1], mode='recursion')
        circuit.rz(phase/2, ancilla[0])
        circuit.mcx(qr2, ancilla[0], ancilla[1], mode='recursion')

def get_driver_component(num_qubits, t, bit_string, use_decompose=True):
    qc = QuantumCircuit(num_qubits)
    apply_convert(qc, bit_string)
    qc.barrier(label='convert')
    qr = qc.qubits[:num_qubits]
    if use_decompose == True:
        ancilla = QuantumRegister(2,name='anc')
        qc.add_register(ancilla)
        decompose_phase_gate(qc, qr, ancilla, -2 * np.pi * t)
    else:
        qc.mcp(-2 * np.pi * t, list(range(1, num_qubits)), 0)
    qc.barrier(label='multi-ctrl')
    qc.x(num_qubits - 1)
    if use_decompose == True:
        decompose_phase_gate(qc, qr, ancilla, 2 * np.pi * t)
    else:
        qc.mcp(2 * np.pi * t, list(range(1, num_qubits)), 0)
    qc.x(num_qubits - 1)
    qc.barrier(label='reverse')
    apply_reverse(qc, bit_string)
    return qc

# 输入qc, 返回电路酉矩阵
def get_circ_unitary(quantum_circuit):
    backend = Aer.get_backend('unitary_simulator' )
    new_circuit = transpile(quantum_circuit, backend)
    job = backend.run(new_circuit)
    result = job.result()
    unitary = result.get_unitary()
    return unitary

def tensor_product(matrices):
    sigma_plus = np.array([[0, 1], [0, 0]])
    sigma_minus = np.array([[0, 0], [1, 0]])
    identity = np.array([[1, 0], [0, 1]])
    mlist = [sigma_plus, sigma_minus, identity]
    result = np.array(mlist[matrices[0]])
    for matrix in matrices[1:]:
        result = np.kron(result, mlist[matrix])
    return result

def get_simulate_unitary(t, bit_string):    
    # for unitary build contrary_v
    U = expm(-1j * 2 * np.pi * t * (tensor_product(bit_string)+tensor_product([not(i) for i in bit_string])))
    return U

def decompose_unitary(t, bit_string):
    unitary = get_simulate_unitary(t,bit_string)
    qc = QuantumCircuit(len(bit_string))
    qc.unitary(unitary, range(len(bit_string)))
    return qc.decompose()

def decompose_circuit(qc, max_control_qubits=5):
    from qiskit import transpile
    control_gates = []
    # do not work
    # for i in range(1, max_control_qubits + 1):
    #     control_gates.append(''.join(['c'] * i + ['z']))
    qc = transpile(qc, basis_gates=[*control_gates, 'h', 'x', 'y', 'cx', 'barrier', 'p'], optimization_level=3)
    return qc

if __name__ == '__main__':
    def set_print_form(suppress=True, precision=4, linewidth=300):
    # 不要截断 是否使用科学计数法 输出浮点数位数 宽度
        np.set_printoptions(threshold=np.inf, suppress=suppress, precision=precision,  linewidth=linewidth)
    set_print_form()
    import numpy as np
    def get_decompose_time(bit_string=[0, 0, 1, 1], f=None):
        t = 0.9
        time_dict = {}
        write_string = ''
        num_qubits = len(bit_string)
        time_start = perf_counter()
        qc = get_driver_component(num_qubits, t, bit_string, False)
        # print(get_simulate_unitary(t, bit_string))
        # print(get_circ_unitary(qc))
        # print(np.allclose(get_circ_unitary(qc), get_simulate_unitary(t, bit_string)))
        # print(qc.draw())
        # exit()
        time_end = perf_counter()
        time_init = time_end - time_start
        time_dict['ours'] = time_init
        print(f'decompose time: {time_end - time_start}')
        depth = qc.depth()
        num_gates = len(qc.data)
        print(f'depth: {depth}')
        print(f'num_gates: {num_gates}')
        write_string += f'{str(num_qubits)}, {str(num_qubits)}, {str(time_init)}, {str(depth)}, {str(num_gates)},\n'

        # decompose into multi-qubit gates
        for i in range(6, 1, -1):
            time_start = perf_counter()
            transpile_circuit = decompose_circuit(qc, i)
            time_end = perf_counter()
            time_decompose = time_end - time_start
            depth = transpile_circuit.depth()
            num_nonlocal_gates = transpile_circuit.num_nonlocal_gates()
            time_decompose += time_init
            print(f'max_control_qubits: {i}')
            print(f'depth: {depth}')
            print(f'num_nonlocal_gates: {num_nonlocal_gates}')
            print(f'time_decompose: {time_decompose}')
            write_string += f'{str(num_qubits)}, {str(i)}, {str(time_decompose)}, {str(depth)}, {str(num_gates)},\n'
        
        time_start = perf_counter()
        uqc = decompose_unitary(t, bit_string)
        time_end = perf_counter()
        print(f'unitary_time: {time_end - time_start}')
        udepth = uqc.depth()
        unum_gates = len(uqc.count_ops())
        print(f'udepth: {udepth}')
        print(f'unum_gates: {unum_gates}') #?? it's wrong
        time_dict['unitary'] = time_end - time_start
        write_string += f'{str(num_qubits)}, {str(2)}, {str(time_dict["unitary"])}, {str(udepth)}, {str(unum_gates)},\n'
        if f:
            f.write(write_string)
        return time_dict
    time_list = []
    with open('../../../implementations/data/decompose_result.csv','w+') as f:
        f.write('num_qubits, ours, unitary\n')
        for i in range(3, 9):
            bit_string = np.random.randint(2, size=i)
            # bit_string = [0,1,0, 1, 1,1,1]
            print(f'num_qubits: {i}')
            time = get_decompose_time(bit_string, f)
            time_list.append(time)
            print('--------------------')
    import pandas as pd
    df = pd.DataFrame(time_list)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    scale = 0.009
    fig = plt.figure(figsize=(2500 * scale , 800 * scale))
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 62
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # set axes linewidth
    mpl.rcParams['axes.linewidth'] = 5
    ## set ticks linewidth
    mpl.rcParams['xtick.major.size'] = 20
    mpl.rcParams['xtick.major.width'] = 5
    mpl.rcParams['ytick.major.size'] = 20
    mpl.rcParams['ytick.major.width'] = 5
    axes = plt.axes([0,0,1,1])
    for i, col in enumerate(df.columns):
        axes.bar(df.index + 0.3 * i, df[col], width=0.3, label=col)
    axes.legend(frameon=False, bbox_to_anchor=(1, 1), ncol=4)
    axes.tick_params(axis='x', which='major', length=20)
    axes.set_xlabel('# qubits')
    axes.set_yscale('log')
    axes.set_xticks(df.index + 0.15)
    axes.set_xticklabels(df.index + 3)
    axes.grid(axis='y', linestyle='--', linewidth=3, color='#B0B0B0')
    axes.set_ylabel('Time (s)')
    fig.savefig('../../../implementations/figures/decompose_optimize.pdf', dpi=600, format='pdf', bbox_inches='tight')