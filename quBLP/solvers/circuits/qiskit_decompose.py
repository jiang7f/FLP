from time import perf_counter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import transpile
from scipy.linalg import expm
import numpy as np
def iter_apply(qc,bitstring,i,num_qubits):
    flip = bitstring[0]==0
    if flip:
        bitstring = [not(j) for j in bitstring]
    if bitstring[1]:
        qc.x(num_qubits-i-1)
    qc.cx(num_qubits - i,num_qubits-(i+1))
    next_bitstring = bitstring[1:]
    if len(next_bitstring)>1:
        return iter_apply(qc,next_bitstring,i+1,num_qubits)
    else:
        return qc

def apply_convert(qc,bitstring):
    num_qubits = len(bitstring)
    qc.x(num_qubits - 1)
    qc.h(num_qubits - 1)
    iter_apply(qc,bitstring,1,num_qubits)

def reversed_iter_apply(qc,bitstring,i,num_qubits):
    flip = bitstring[0]==0
    if flip:
        bitstring = [not(j) for j in bitstring]
    if len(bitstring)>1:
        reversed_iter_apply(qc,bitstring[1:],i+1,num_qubits)
    else:
        return qc
    qc.cx(num_qubits - i,num_qubits-(i+1))
    if bitstring[1]:
        qc.x(num_qubits-i-1)
    return qc
    

def reverse_apply_convert(qc,bitstring):
    ## 1111
    num_qubits = len(bitstring)
    reversed_iter_apply(qc,bitstring,1,num_qubits)
    qc.h(num_qubits - 1)
    qc.x(num_qubits - 1)
    ## 1111+0111
    ##find bit start with 1


def decompose_phase_gate(circuit:QuantumCircuit,qr:QuantumRegister,ancilla:QuantumRegister,phase:float)->QuantumCircuit:
    """
    Decompose a phase gate into a series of controlled-phase gates.
    Args:
        phase_num_qubit (int): the number of qubits that the phase gate acts on.
        phase (float): the phase angle of the phase gate.
        max_num_qubit_control (int): the maximum number of qubits that the controlled-phase gates can control.
    Returns:
        QuantumCircuit: the circuit that implements the decomposed phase gate.
    """
    phase_num_qubit = len(qr)
    if phase_num_qubit == 1:
        circuit.p(phase,0)
    elif phase_num_qubit == 2:
        circuit.cp(phase,0,1)
    else:
        ## convert into the multi-cx gate 
        ## partition qubits into two sets
        half_num_qubit = phase_num_qubit//2
        qr1 = qr[:half_num_qubit]
        qr2 = qr[half_num_qubit:]
        circuit.rz(-phase/2,ancilla[0])
        circuit.mcx(qr1,ancilla[0],ancilla[1],mode='recursion')
        circuit.rz(phase/2,ancilla[0])
        circuit.mcx(qr2,ancilla[0],ancilla[1],mode='recursion')
        circuit.rz(-phase/2,ancilla[0])
        circuit.mcx(qr1,ancilla[0],ancilla[1],mode='recursion')
        circuit.rz(phase/2,ancilla[0])
        circuit.mcx(qr2,ancilla[0],ancilla[1],mode='recursion')



def get_driver_component(num_qubits,t,bitstring,use_decompose=True):
    qc = QuantumCircuit(num_qubits)
    ancilla = QuantumRegister(2,name='anc')
    qc.add_register(ancilla)
    reverse_apply_convert(qc,bitstring)
    qc.barrier(label='convert')
    qr = qc.qubits[:num_qubits]
    if use_decompose:
        decompose_phase_gate(qc,qr,ancilla,-2*np.pi * t)
    else:
        qc.mcp(-2*np.pi * t, list(range(1,num_qubits)),0)
    
    qc.barrier(label='multi-ctrl')
    qc.x(num_qubits-1)
    if use_decompose:
        decompose_phase_gate(qc,qr,ancilla,2*np.pi * t)
    else:
        qc.mcp(2*np.pi * t, list(range(1,num_qubits)),0)
    qc.x(num_qubits-1)
    qc.barrier(label='reverse')
    # qc.mcp(-2*np.pi * t, list(range(num_qubits-1)),num_qubits-1)
    apply_convert(qc,bitstring)
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

def get_simulate_unitary(t,bitstring):    
    # for unitary build contrary_v
    U = expm(-1j * 2 * np.pi * t * (tensor_product(bitstring)+tensor_product([not(i) for i in bitstring])))
    return U

def decompose_unitary(t,bitstring):
    unitary = get_simulate_unitary(t,bitstring)
    qc = QuantumCircuit(len(bitstring))
    qc.unitary(unitary,range(len(bitstring)))
    return qc.decompose()

def decompose_circuit(qc,max_control_qubits=5):
    from qiskit import transpile
    control_gates =[]
    for i in range(1,max_control_qubits+1):
        control_gates.append(''.join(['c']*i+['z']))
    qc = transpile(qc, basis_gates=[''.join(['c']*max_control_qubits+['z']),'h','x','y','cx','barrier','p'],optimization_level=3)
    return qc

if __name__ == '__main__':
    import numpy as np
    def get_decompose_time(bitstring= [0,0,1,1],f=None):
        t = 0.9
        time = {}
        write_string = ''
        num_q = len(bitstring)
        start = perf_counter()
        qc =get_driver_component(num_q,t,bitstring)
        end = perf_counter()
        initime = end-start
        time['ours'] = initime
        print('decompose time:',end-start)
        depth = qc.depth()
        num_gates = len(qc.data)
        print('depth:',depth)
        print('num_gates:',num_gates)
        write_string += str(num_q)+','+str(num_q)+','+str(initime)+','+str(depth)+','+str(num_gates)+',\n'
        ## decompose into multi-qubit gates
        start = perf_counter()
        for i in range(6,1,-1):
            start = perf_counter()
            transpile_circuit = decompose_circuit(qc,i)
            end = perf_counter()
            dtime = end-start
            depth = transpile_circuit.depth()
            num_gates = transpile_circuit.num_nonlocal_gates()
            dtime += initime
            print('max_control_qubits:',i)
            print('depth:',depth)
            print('num_gates:',num_gates)
            print('decompose time:',dtime)
            write_string += str(num_q)+','+str(i)+','+str(dtime)+','+str(depth)+','+str(num_gates)+',\n'
        
        start = perf_counter()
        uqc = decompose_unitary(t,bitstring)
        end = perf_counter()
        print('unitary time:',end-start)
        udepth = uqc.depth()
        unum_gates = len(qc.count_ops())
        print('udepth:',udepth)
        print('unum_gates:',unum_gates)
        time['unitary'] = end-start
        write_string += str(num_q)+','+str(2)+','+str(time['unitary'])+','+str(udepth)+','+str(unum_gates)+',\n'
        if f:
            f.write(write_string)
        return time
    times = []
    with open('implementations/data/decompose_result.csv','w') as f:
        f.write('num_qubits,ours,unitary\n')
        for i in range(3,9):
            bitstring = np.random.randint(2,size=i)
            print('num_qubits:',i)
            time = get_decompose_time(bitstring,f)
            times.append(time)
            print('-----------------')
    ## plot the time
    import pandas as pd
    df = pd.DataFrame(times)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    scale = 0.009
    fig = plt.figure(figsize=(2500*scale , 800*scale))
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
    for i,col in enumerate(df.columns):
        axes.bar(df.index+0.3*i,df[col],width=0.3,label=col)
    axes.legend(frameon=False,bbox_to_anchor=(2.5,1.3),ncol=4)
    axes.tick_params(axis='x',which='major',length=20)
    # axes.set_xticklabels(allnames,rotation=90)
    axes.set_xlabel('# qubits')
    axes.set_yscale('log')
    axes.set_xticks(df.index+0.15)
    axes.set_xticklabels(df.index+3)
    axes.grid(axis='y',linestyle='--',linewidth=3,color='#B0B0B0')
    axes.set_ylabel('Time (s)')
    fig.savefig('implementations/figures/decompose_optimize.pdf',dpi=600,format='pdf',bbox_inches='tight')

    
