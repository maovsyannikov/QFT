import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from collections import Counter, OrderedDict
import functools
pi = np.pi

from qiskit import BasicAer, IBMQ
from qiskit import QuantumCircuit, execute
%config InlineBackend.figure_format = 'svg'

from qiskit.tools.monitor import job_monitor

from qiskit.visualization import plot_circuit_layout
from qiskit.visualization.counts_visualization import _plot_histogram_data

def qft_rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cu1(pi/2**(n-qubit), qubit, n)
    qft_rotations(circuit, n)
    
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def graph(plot,figsize=(9,6)):
    legend=None
    if isinstance(plot, dict):
        plot = [plot] 
    width = 1/(len(plot)+1) 
    rects=[]
    labels = list(sorted(functools.reduce(lambda x, y: x.union(y.keys()), plot, set())))
    labels_dict, all_pvalues, all_inds = _plot_histogram_data(plot,labels,None)
    all_pvalues=[i*100 for i in all_pvalues]
    fig, ax = plt.subplots(figsize=figsize)
    for item, _ in enumerate(plot):
            for idx, val in enumerate(all_pvalues[item]):
                label = None
                if not idx and legend:
                    label = legend[item]
                if val >= 0:
                        rects.append(ax.bar(idx, val,width,color='#648fff',label=label, zorder=2))
            ax.set_xticks(all_inds[item])
            ax.set_xticklabels(labels_dict.keys(), fontsize=10, rotation=90)
            for rect in rects:
                    for rec in rect:
                        height = rec.get_height()
                        ax.text(rec.get_x() + rec.get_width() / 2., 1.02* height,
                                    '%.6f' % float(height),
                                    ha='center', va='bottom',rotation=90, zorder=3)
    ax.set_ylabel('Probabilities', fontsize=12)
    ax.set_xlabel('States', fontsize=12)
    all_vals = np.concatenate(all_pvalues).ravel()
    ax.set_ylim([0, min([100, max([val+2.5 for val in all_vals])])])
    return fig
    
n = 5
qft_circuit = QuantumCircuit(n)
circuit = QuantumCircuit(1)
circuit.h(0)
qft(qft_circuit, n)
qft_circuit.measure_all()
qft_circuit.draw(output='mpl')

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend=provider.get_backend('ibmq_london')
qc_basis = qft_circuit.decompose()
qqq = transpile(qc_basis, backend=backend, initial_layout=[0,2,4,3,1])
shots = 1024
job_exp = execute(qc_basis, backend=backend_lond, shots=shots)
job_monitor(job_exp)
results = job_exp.result()
dict1=results.get_counts()
fig=graph(dict1)
