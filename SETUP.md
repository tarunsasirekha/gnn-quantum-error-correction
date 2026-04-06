# Setup Guide for GNN Quantum Error Correction

## Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA toolkit (optional, for GPU acceleration)

## Step-by-Step Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv qec_env

# Activate it
# On Linux/Mac:
source qec_env/bin/activate
# On Windows:
qec_env\Scripts\activate
```

### 2. Install PyTorch

Visit https://pytorch.org/get-started/locally/ and select your configuration.

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch Geometric

PyTorch Geometric requires special installation steps:

```bash
pip install torch-geometric
```

If you encounter issues, try:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

Replace `cpu` with your CUDA version if using GPU (e.g., `cu118` for CUDA 11.8).

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- numpy
- scipy
- matplotlib
- networkx
- tqdm

## Verify Installation

Run the test script:

```bash
python test_all.py
```

You should see:
```
============================================================
RUNNING TESTS FOR GNN QUANTUM ERROR CORRECTION
============================================================
Testing imports...
  ✓ Core dependencies
  ✓ All modules

Testing code generation...
  ✓ Toric code generation

Testing error simulation...
  ✓ Error simulation

Testing graph creation...
  ✓ Graph creation

Testing GNN model...
  ✓ GNN model forward pass

Testing training...
  ✓ Training loop

Testing classical decoders...
  ✓ Classical decoders

Testing evaluation...
  ✓ Evaluation

============================================================
TEST SUMMARY
============================================================
Passed: 8/8

✓ ALL TESTS PASSED! System is ready to use.
```

## Troubleshooting

### PyTorch Geometric Installation Issues

**Problem**: "No matching distribution found for torch-geometric"

**Solution**: Install from source or use conda:
```bash
conda install pyg -c pyg
```

**Problem**: "undefined symbol" errors

**Solution**: Ensure PyTorch and PyG versions match. Reinstall both:
```bash
pip uninstall torch torch-geometric
# Then reinstall following steps 2-3
```

### CUDA Issues

**Problem**: "CUDA out of memory"

**Solutions**:
1. Use CPU mode (don't use `--use-cuda` flag)
2. Reduce batch size or model size
3. Use smaller quantum codes

**Problem**: "CUDA not available" when you have a GPU

**Solution**: Reinstall PyTorch with correct CUDA version:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

**Problem**: "ModuleNotFoundError: No module named 'torch_geometric'"

**Solution**: PyTorch Geometric not installed. Follow step 3 above.

**Problem**: "ImportError: cannot import name 'MessagePassing'"

**Solution**: Incompatible PyG version. Reinstall:
```bash
pip install --upgrade torch-geometric
```

### Performance Issues

**Slow training:**
1. Use GPU with `--use-cuda`
2. Reduce model size: `--hidden-dim 32 --num-layers 3`
3. Use fewer training samples initially

**Out of memory:**
1. Reduce code size: `--code-size 3`
2. Reduce batch size (modify training.py)
3. Use smaller model

## Quick Test

After installation, run a quick demo:

```bash
python main.py --code-size 3 --n-train 100 --n-test 50 --epochs 5
```

This should complete in 1-2 minutes and show decoder comparison results.

## Platform-Specific Notes

### Linux
- Should work out of the box
- For GPU, ensure NVIDIA drivers are installed
- May need `sudo apt-get install python3-dev` for some dependencies

### macOS
- PyTorch CPU works well on M1/M2 chips
- Use `pip install torch` (no CUDA needed)
- May need Xcode command line tools: `xcode-select --install`

### Windows
- Use Anaconda for easier dependency management
- PowerShell may need execution policy change for venv
- CUDA support available with proper drivers

## Conda Alternative

If pip installation fails, try conda:

```bash
# Create conda environment
conda create -n qec_env python=3.10
conda activate qec_env

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyG
conda install pyg -c pyg

# Install other dependencies
pip install matplotlib networkx tqdm
```

## Docker Alternative

For reproducible environment:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install torch-geometric matplotlib networkx tqdm scipy

COPY . /workspace
WORKDIR /workspace

CMD ["python", "main.py"]
```

## Minimal Installation (No GPU)

If you just want to test classical decoders or do small experiments:

```bash
pip install numpy scipy matplotlib
# Skip PyTorch installation
# Only qldpc_codes.py, error_simulation.py, and classical_decoders.py will work
```

## Development Setup

For development and debugging:

```bash
pip install jupyter ipython pytest black flake8

# Run Jupyter notebook
jupyter notebook

# Format code
black *.py

# Run linter
flake8 *.py
```

## Next Steps

Once installation is complete:

1. Read the README.md for usage examples
2. Run `python test_all.py` to verify everything works
3. Try the quick demo: `python main.py`
4. Experiment with different parameters
5. Start your research project!

## Getting Help

- Check README.md for detailed documentation
- Run `python main.py --help` for command-line options
- Each module can be tested individually: `python qldpc_codes.py`
- PyTorch docs: https://pytorch.org/docs/
- PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io/

## System Requirements

**Minimum:**
- 4GB RAM
- Dual-core CPU
- 1GB disk space

**Recommended:**
- 8GB+ RAM
- Quad-core CPU or better
- NVIDIA GPU with 4GB+ VRAM (for larger codes)
- 5GB disk space

**For FPGA work later:**
- Xilinx Vitis/Vivado or Intel Quartus
- FPGA board (Zynq, Artix-7, etc.)
- Additional 20GB+ disk space for tools
