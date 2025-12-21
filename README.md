# QML Barren Plateaus

Empirical investigation of the barren plateau phenomenon in parameterized quantum circuits. Studies how gradient variance scales with circuit width and depth, and evaluates strategies for maintaining trainability in variational quantum machine learning models.

Built as part of independent study work on quantum-classical hybrid optimization.

## Background

### What Are Barren Plateaus?

A **barren plateau** occurs when the gradient of the cost function vanishes exponentially with the number of qubits. In a barren plateau landscape, the cost function looks essentially flat everywhere — the optimizer can't find a direction to improve, regardless of where it starts.

Formally, for a parameterized circuit $U(\theta)$ and cost function $C(\theta) = \langle 0 | U^\dagger(\theta) \, O \, U(\theta) | 0 \rangle$:

$$\text{Var}_\theta\left[\frac{\partial C}{\partial \theta_k}\right] \leq F(n)$$

where $F(n)$ decreases **exponentially** with the number of qubits $n$ for certain circuit architectures. This means the gradient is exponentially close to zero with exponentially high probability.

### Why This Matters for QML

Barren plateaus are arguably the most significant obstacle for scaling variational quantum algorithms. If your quantum ML model has a barren plateau, it's essentially untrainable — and the problem gets worse as you add more qubits. This project empirically investigates:

1. Which circuit architectures suffer from barren plateaus
2. Whether global vs local cost functions make a difference
3. What initialization and training strategies can help

### Theoretical Background

Key results from the literature:
- **McClean et al. (2018)**: Random parameterized circuits have barren plateaus
- **Cerezo et al. (2021)**: Local cost functions can avoid barren plateaus for shallow circuits
- **Grant et al. (2019)**: Identity-block initialization can provide non-vanishing gradients

## Project Structure

```
qml-barren-plateaus/
├── src/
│   ├── circuits.py          # Parameterized circuit construction
│   ├── gradients.py         # Parameter shift rule and variance computation
│   ├── cost_functions.py    # Global and local observables
│   ├── trainability.py      # Layer-wise training and initialization
│   └── plotting.py          # Visualization functions
├── notebooks/
│   └── barren_plateau_analysis.ipynb
├── results/
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/qml-barren-plateaus.git
cd qml-barren-plateaus
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.circuits import build_hardware_efficient_ansatz
from src.gradients import compute_gradient_variance
from src.cost_functions import global_cost_observable

circuit = build_hardware_efficient_ansatz(n_qubits=6, depth=4)
observable = global_cost_observable(n_qubits=6)

variance = compute_gradient_variance(
    circuit, observable, param_index=0, n_samples=200
)
print(f"Gradient variance: {variance:.6e}")
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/barren_plateau_analysis.ipynb
```

## Results

### Gradient Variance vs Number of Qubits

For the hardware-efficient ansatz with global cost function (depth=4):

| Qubits | Gradient Variance | Scale |
|--------|------------------|-------|
| 2 | 1.2e-01 | Trainable |
| 4 | 3.1e-02 | Trainable |
| 6 | 4.7e-03 | Marginal |
| 8 | 6.2e-04 | Barren |
| 10 | 8.1e-05 | Barren |
| 12 | 9.3e-06 | Barren |

*Averaged over 200 random parameter initializations.*

### Key Findings

- **Exponential decay confirmed**: Hardware-efficient ansatz with global cost shows clear exponential gradient variance decay, consistent with McClean et al.
- **Local cost helps**: Switching from global to local cost function increases gradient variance by 1-2 orders of magnitude at 10+ qubits
- **Identity initialization works**: Starting near identity preserves gradient information for the first few optimization steps
- **Layer-wise training is effective**: Building circuits incrementally avoids the deep-circuit barren plateau by never optimizing all parameters at once
- **Structured ansatze are better**: Limited entanglement patterns show slower gradient decay than fully random architectures

## References

1. McClean, J. R., et al. (2018). "Barren plateaus in quantum neural network training landscapes." *Nature Communications*, 9, 4812.
2. Cerezo, M., et al. (2021). "Cost function dependent barren plateaus in shallow parametrized quantum circuits." *Nature Communications*, 12, 1791.
3. Grant, E., et al. (2019). "An initialization strategy for addressing barren plateaus in parametrized quantum circuits." *Quantum*, 3, 214.
4. Pesah, A., et al. (2021). "Absence of Barren Plateaus in Quantum Convolutional Neural Networks." *Physical Review X*, 11(4), 041011.

## License

MIT License — see [LICENSE](LICENSE) for details.
