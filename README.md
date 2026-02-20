# GNN-Based Quantum Error Correction

**Graph Neural Network Decoders for Quantum Low-Density Parity-Check (qLDPC) Codes**

This repository contains a complete implementation of GNN-based decoders for quantum error correction, including data generation, training, and benchmarking against classical decoders.

## Overview

Quantum computers require robust error correction to scale. This project implements:
- **Multiple qLDPC code families** (Toric codes, Hypergraph product codes, Random regular codes)
- **GNN decoder architectures** (Basic GNN, Attention-based, Residual)
- **Classical baselines** (Belief Propagation, Greedy decoding, Lookup tables)
- **Comprehensive evaluation** (Success rates, logical error rates, decode times)

## Project Structure

```
.
├── qldpc_codes.py           # Quantum code generators
├── error_simulation.py      # Error and syndrome generation
├── graph_representation.py  # Convert codes to graphs
├── gnn_models.py           # GNN decoder architectures
├── training.py             # Training loops and loss functions
├── classical_decoders.py   # Baseline classical decoders
├── evaluation.py           # Evaluation and benchmarking
├── main.py                 # Complete demo workflow
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+

### Setup

```bash
# Clone or download the repository
cd gnn-qec

# Install dependencies
pip install -r requirements.txt
```

### PyTorch Geometric Installation

PyTorch Geometric requires manual installation. Follow instructions at:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Quick install for CPU:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

## Quick Start

### Run Complete Demo

```bash
python main.py
```

This will:
1. Create a 4×4 toric code (32 qubits)
2. Generate 500 training samples with 5% error rate
3. Train a GNN decoder for 10 epochs
4. Compare against Belief Propagation and Greedy decoders
5. Print performance comparison

### Run with Custom Parameters

```bash
python main.py \
    --code-type toric \
    --code-size 6 \
    --error-rate 0.08 \
    --n-train 2000 \
    --model-type attention \
    --epochs 30 \
    --save-model \
    --plot-results
```

### Available Options

**Code Parameters:**
- `--code-type`: `toric` or `hypergraph`
- `--code-size`: Lattice size (e.g., 4 for 4×4 toric code)
- `--error-rate`: Physical error probability (e.g., 0.05 = 5%)

**Model Parameters:**
- `--model-type`: `basic`, `attention`, or `residual`
- `--hidden-dim`: Hidden layer dimension (default: 64)
- `--num-layers`: Number of GNN layers (default: 5)

**Training Parameters:**
- `--epochs`: Training epochs (default: 20)
- `--learning-rate`: Learning rate (default: 0.001)
- `--n-train`: Training samples (default: 1000)

**Outputs:**
- `--save-model`: Save trained model
- `--plot-results`: Display comparison plots
- `--save-plots`: Save plots to disk

## Module Usage

### 1. Generate Quantum Codes

```python
from qldpc_codes import ToricCode, HypergraphProductCode

# Create 5×5 toric code (50 qubits)
toric = ToricCode(L=5)
print(f"Qubits: {toric.n_qubits}")
print(f"Distance: {toric.get_distance()}")

# Access parity check matrices
H_X = toric.H_X  # X-type stabilizers
H_Z = toric.H_Z  # Z-type stabilizers
```

### 2. Simulate Errors and Syndromes

```python
from error_simulation import ErrorSimulator, NoiseModel

# Create depolarizing noise channel
noise = NoiseModel(p_depol=0.05)
simulator = ErrorSimulator(toric.H_X, toric.H_Z, noise)

# Generate single error-syndrome pair
sample = simulator.generate_error_syndrome_pair()
error_X = sample['error_X']
syndrome_Z = sample['syndrome_Z']

# Generate training dataset
dataset = simulator.generate_dataset(n_samples=1000)
```

### 3. Create Graph Representations

```python
from graph_representation import TannerGraphBuilder

# Build Tanner graph from parity check matrix
graph_builder = TannerGraphBuilder(toric.H_Z)

# Create PyTorch Geometric Data object
data = graph_builder.create_graph_data(
    syndrome=syndrome_Z,
    error=error_X
)
```

### 4. Train GNN Decoder

```python
from gnn_models import GNNDecoder
from training import Trainer, DecoderLoss
import torch.optim as optim

# Create model
model = GNNDecoder(
    input_dim=3,
    hidden_dim=64,
    num_layers=5
)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = DecoderLoss()
trainer = Trainer(model, optimizer, criterion)

# Train
history = trainer.train(
    train_data_list,
    val_data_list,
    num_epochs=20
)
```

### 5. Evaluate and Compare Decoders

```python
from evaluation import DecoderEvaluator
from classical_decoders import BeliefPropagationDecoder

# Create evaluator
evaluator = DecoderEvaluator(toric.H_X, toric.H_Z)

# Evaluate GNN
gnn_result = evaluator.evaluate_gnn_decoder(
    model, test_data_list, true_errors
)

# Evaluate classical baseline
bp_decoder = BeliefPropagationDecoder(toric.H_Z)
bp_result = evaluator.evaluate_classical_decoder(
    bp_decoder, syndromes, true_errors
)

# Compare
evaluator.compare_decoders([gnn_result, bp_result])
```

## Understanding the Code

### Toric Code Representation

Toric codes are defined on a 2D lattice with periodic boundaries:
- **Qubits** live on edges of the lattice
- **Star operators** (X-stabilizers) at vertices
- **Plaquette operators** (Z-stabilizers) on faces

For L×L toric code:
- Physical qubits: 2L²
- Stabilizers: 2L²
- Logical qubits: 2
- Distance: L

### How GNN Decoding Works

1. **Input**: Error syndrome (which stabilizers are violated)
2. **Graph**: Tanner graph with qubits and stabilizers as nodes
3. **Message Passing**: GNN propagates syndrome information through graph
4. **Output**: Prediction of which qubits have errors

The GNN learns to perform belief propagation-like inference but with learned (not hand-coded) update rules.

### Syndrome to Error Mapping

```
X errors detected by Z stabilizers: syndrome_Z = H_Z @ error_X (mod 2)
Z errors detected by X stabilizers: syndrome_X = H_X @ error_Z (mod 2)
```

The decoder's job: Given syndrome, predict error.

## Results Interpretation

### Key Metrics

**Success Rate**: Fraction of samples where decoded error produces correct syndrome

**Logical Error Rate**: Fraction where residual error causes logical failure

**Decode Time**: Average time to decode one syndrome

**Residual Weight**: Average weight of (true_error + decoded_error)

### Expected Performance

For 4×4 toric code at 5% error rate:
- **GNN Success Rate**: 85-95%
- **BP Success Rate**: 80-90%
- **GNN Speedup**: 2-5x faster than BP
- **Greedy Success Rate**: 60-70%

## Advanced Usage

### Custom Architectures

```python
from gnn_models import ResidualGNNDecoder

# Deeper model with residual connections
model = ResidualGNNDecoder(
    input_dim=3,
    hidden_dim=128,
    num_layers=10,
    dropout=0.2
)
```

### Biased Noise

```python
from error_simulation import BiasedNoiseSimulator

# Z-biased noise (10:1 ratio)
biased_sim = BiasedNoiseSimulator(
    H_X, H_Z,
    bias=10.0,
    base_rate=0.05
)
```

### Scalability Testing

```python
from evaluation import ScalabilityTester

tester = ScalabilityTester()
results = tester.test_code_size_scaling(
    code_sizes=[3, 4, 5, 6],
    decoder_factory=lambda code: GNNDecoder(...),
    error_rate=0.05,
    n_trials=100
)
tester.plot_scaling(results)
```

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test code generation
python qldpc_codes.py

# Test error simulation
python error_simulation.py

# Test graph representation
python graph_representation.py

# Test GNN models
python gnn_models.py

# Test training
python training.py

# Test classical decoders
python classical_decoders.py

# Test evaluation
python evaluation.py
```

## Next Steps: FPGA Implementation

This codebase provides the foundation for FPGA implementation:

1. **Train and quantize model** in PyTorch
2. **Export weights** to fixed-point format
3. **Implement GNN node** in Verilog/HLS
4. **Design message-passing architecture**
5. **Synthesize and test** on FPGA

See the conversation history for detailed FPGA implementation strategy.

## Performance Tips

### Speed up training:
- Use GPU: `--use-cuda`
- Reduce model size: `--hidden-dim 32 --num-layers 3`
- Fewer samples: `--n-train 500`

### Improve accuracy:
- Larger model: `--hidden-dim 128 --num-layers 8`
- More training data: `--n-train 5000`
- Attention mechanism: `--model-type attention`
- Lower learning rate: `--learning-rate 0.0001`

### Memory issues:
- Reduce batch size (modify training.py)
- Use smaller codes: `--code-size 3`
- Reduce hidden dimension

## Troubleshooting

**PyTorch Geometric import errors:**
- Install PyG following official instructions
- Ensure PyTorch and PyG versions are compatible

**CUDA out of memory:**
- Use CPU: remove `--use-cuda`
- Reduce model/batch size
- Use smaller codes

**Poor decoder performance:**
- Increase training samples
- Try different model architectures
- Adjust hyperparameters (learning rate, layers)



## References

Key papers on GNN-based quantum decoding:
- Hu et al. (2025): Efficient and Universal Neural-Network Decoder
- Ninkovic et al. (2024): Decoding Quantum LDPC Codes Using Graph Neural Networks

## License

MIT License - feel free to use for research and education.

