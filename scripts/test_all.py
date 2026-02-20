"""
Quick Test Script
Verifies all modules are working correctly
"""

import sys

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    try:
        import torch
        import numpy as np
        from torch_geometric.data import Data
        print("  ✓ Core dependencies")
        
        import qldpc_codes
        import error_simulation
        import graph_representation
        import gnn_models
        import training
        import classical_decoders
        import evaluation
        print("  ✓ All modules")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_code_generation():
    """Test quantum code generation"""
    print("\nTesting code generation...")
    try:
        from qldpc_codes import ToricCode
        
        toric = ToricCode(L=3)
        assert toric.n_qubits == 18, "Wrong number of qubits"
        assert toric.H_X.shape == (9, 18), "Wrong H_X shape"
        assert toric.H_Z.shape == (9, 18), "Wrong H_Z shape"
        
        print("  ✓ Toric code generation")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_error_simulation():
    """Test error simulation"""
    print("\nTesting error simulation...")
    try:
        from qldpc_codes import ToricCode
        from error_simulation import ErrorSimulator, NoiseModel
        
        toric = ToricCode(L=3)
        simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
        
        # Generate single sample
        sample = simulator.generate_error_syndrome_pair()
        assert 'error_X' in sample
        assert 'syndrome_Z' in sample
        
        # Generate dataset
        dataset = simulator.generate_dataset(n_samples=10)
        assert dataset['error_X'].shape == (10, 18)
        
        print("  ✓ Error simulation")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_graph_creation():
    """Test graph representation"""
    print("\nTesting graph creation...")
    try:
        from qldpc_codes import ToricCode
        from error_simulation import ErrorSimulator, NoiseModel
        from graph_representation import TannerGraphBuilder
        import numpy as np
        
        toric = ToricCode(L=3)
        simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
        sample = simulator.generate_error_syndrome_pair()
        
        builder = TannerGraphBuilder(toric.H_Z)
        data = builder.create_graph_data(
            syndrome=sample['syndrome_Z'],
            error=sample['error_X']
        )
        
        assert data.num_nodes == toric.n_qubits + toric.H_Z.shape[0]
        assert data.x.shape[0] == data.num_nodes
        
        print("  ✓ Graph creation")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_gnn_model():
    """Test GNN model"""
    print("\nTesting GNN model...")
    try:
        from qldpc_codes import ToricCode
        from error_simulation import ErrorSimulator, NoiseModel
        from graph_representation import TannerGraphBuilder
        from gnn_models import GNNDecoder
        import torch
        
        toric = ToricCode(L=3)
        simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
        sample = simulator.generate_error_syndrome_pair()
        
        builder = TannerGraphBuilder(toric.H_Z)
        data = builder.create_graph_data(
            syndrome=sample['syndrome_Z'],
            error=sample['error_X']
        )
        
        model = GNNDecoder(input_dim=3, hidden_dim=32, num_layers=3)
        
        with torch.no_grad():
            output = model(data)
        
        assert output.shape == (toric.n_qubits, 1)
        
        print("  ✓ GNN model forward pass")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_training():
    """Test training loop"""
    print("\nTesting training...")
    try:
        from qldpc_codes import ToricCode
        from error_simulation import ErrorSimulator, NoiseModel
        from graph_representation import BatchGraphBuilder
        from gnn_models import GNNDecoder
        from training import Trainer, DecoderLoss
        import torch.optim as optim
        
        toric = ToricCode(L=3)
        simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
        
        # Small datasets
        train_dataset = simulator.generate_dataset(n_samples=20)
        val_dataset = simulator.generate_dataset(n_samples=5)
        
        batch_builder = BatchGraphBuilder(toric.H_X, toric.H_Z)
        train_data_X, train_data_Z = batch_builder.create_batch(train_dataset)
        val_data_X, val_data_Z = batch_builder.create_batch(val_dataset)
        
        model = GNNDecoder(input_dim=3, hidden_dim=16, num_layers=2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = DecoderLoss()
        
        trainer = Trainer(model, optimizer, criterion, device='cpu')
        history = trainer.train(train_data_Z, val_data_Z, num_epochs=2, verbose=False)
        
        assert len(history['train_losses']) == 2
        
        print("  ✓ Training loop")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_classical_decoders():
    """Test classical decoders"""
    print("\nTesting classical decoders...")
    try:
        from qldpc_codes import ToricCode
        from error_simulation import ErrorSimulator, NoiseModel
        from classical_decoders import BeliefPropagationDecoder, GreedyDecoder
        import numpy as np
        
        toric = ToricCode(L=3)
        simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
        sample = simulator.generate_error_syndrome_pair()
        
        # Test BP
        bp_decoder = BeliefPropagationDecoder(toric.H_Z, max_iterations=20)
        bp_result = bp_decoder.decode(sample['syndrome_Z'])
        assert bp_result.shape == (toric.n_qubits,)
        
        # Test Greedy
        greedy_decoder = GreedyDecoder(toric.H_Z)
        greedy_result = greedy_decoder.decode(sample['syndrome_Z'])
        assert greedy_result.shape == (toric.n_qubits,)
        
        print("  ✓ Classical decoders")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_evaluation():
    """Test evaluation module"""
    print("\nTesting evaluation...")
    try:
        from qldpc_codes import ToricCode
        from error_simulation import ErrorSimulator, NoiseModel
        from classical_decoders import GreedyDecoder
        from evaluation import DecoderEvaluator
        
        toric = ToricCode(L=3)
        simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
        dataset = simulator.generate_dataset(n_samples=10)
        
        evaluator = DecoderEvaluator(toric.H_X, toric.H_Z)
        decoder = GreedyDecoder(toric.H_Z)
        
        result = evaluator.evaluate_classical_decoder(
            decoder,
            dataset['syndrome_Z'],
            dataset['error_X'],
            name="Test"
        )
        
        assert 0 <= result.success_rate <= 1
        
        print("  ✓ Evaluation")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("RUNNING TESTS FOR GNN QUANTUM ERROR CORRECTION")
    print("="*60)
    
    tests = [
        test_imports,
        test_code_generation,
        test_error_simulation,
        test_graph_creation,
        test_gnn_model,
        test_training,
        test_classical_decoders,
        test_evaluation
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED! System is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
