# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.circuit import Instruction
from qiskit.circuit.library import CDKMRippleCarryAdder

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
from qiskit_ibm_provider import least_busy

print("qiskit-imports", end="")

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

print(", qiskit-runtime-service", end="")

# Invoke a primitive. For more details see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials.html
# result = Sampler("ibmq_qasm_simulator").run(circuits).result()

import numpy as np

from dataclasses import dataclass
from math import pi, cos, sin, tan, acos, asin, atan, atan2, log, log2, sqrt

print(", misc-imports", end="")

def get_least_busy_backend():
    return least_busy(service.backends(
        filters=lambda b:                   \
        True                                \
        and b.configuration().n_qubits >= 1 \
        and not b.configuration().simulator \
        and b.status().operational==True
    ))

def simulate_quantum_circuit(circuit, shots=1024):
    simulator = Aer.get_backend("aer_simulator")
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts

print(", and helper-functions", end="")

print(" have been loaded. ")
