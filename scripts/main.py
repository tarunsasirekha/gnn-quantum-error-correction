"""
Main Demo Script for GNN-based Quantum Error Correction
Complete workflow: data generation -> training -> evaluation
"""

import torch
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
import time

from qldpc_codes import ToricCode, HypergraphProductCode, get_classical_repetition_code
from error_simulation import ErrorSimulator, NoiseModel
from graph_representation import BatchGraphBuilder
from gnn_models import GNNDecoder, ResidualGNNDecoder, count_parameters
from training import Trainer, DecoderLoss
from classical_decoders import BeliefPropagationDecoder, GreedyDecoder
from evaluation import DecoderEvaluator
from experiment_logger import ExperimentLogger


def setup_experiment(args):
    """Setup code, simulator, and data"""
    print("="*80)
    print("EXPERIMENT SETUP")
    print("="*80)
    
    # Create quantum code
    if args.code_type == 'toric':
        print(f"\nCreating Toric Code with L={args.code_size}")
        code = ToricCode(L=args.code_size)
    elif args.code_type == 'hypergraph':
        print(f"\nCreating Hypergraph Product Code")
        H_classical = get_classical_repetition_code(args.code_size)
        code = HypergraphProductCode(H_classical, H_classical)
    else:
        raise ValueError(f"Unknown code type: {args.code_type}")
    
    print(f"  Qubits: {code.n_qubits}")
    print(f"  X-checks: {code.H_X.shape[0]}")
    print(f"  Z-checks: {code.H_Z.shape[0]}")
    if hasattr(code, 'get_distance'):
        print(f"  Distance: {code.get_distance()}")
    
    # Create error simulator
    print(f"\nSetting up noise model (p_error = {args.error_rate})")
    noise_model = NoiseModel(p_depol=args.error_rate)
    simulator = ErrorSimulator(code.H_X, code.H_Z, noise_model)
    
    return code, simulator


def generate_datasets(simulator, args):
    """Generate training, validation, and test datasets"""
    print("\n" + "="*80)
    print("GENERATING DATASETS")
    print("="*80)
    
    print(f"\nGenerating {args.n_train} training samples...")
    train_dataset = simulator.generate_dataset(n_samples=args.n_train)
    
    print(f"Generating {args.n_val} validation samples...")
    val_dataset = simulator.generate_dataset(n_samples=args.n_val)
    
    print(f"Generating {args.n_test} test samples...")
    test_dataset = simulator.generate_dataset(n_samples=args.n_test)
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Avg error weight: {train_dataset['error_X'].sum(axis=1).mean():.2f}")
    print(f"  Avg syndrome weight: {train_dataset['syndrome_Z'].sum(axis=1).mean():.2f}")
    
    return train_dataset, val_dataset, test_dataset


def create_graph_data(code, datasets):
    """Convert datasets to graph format"""
    print("\n" + "="*80)
    print("CREATING GRAPH REPRESENTATIONS")
    print("="*80)
    
    train_dataset, val_dataset, test_dataset = datasets
    
    batch_builder = BatchGraphBuilder(code.H_X, code.H_Z)
    
    print("\nConverting to graphs...")
    train_data_X, train_data_Z = batch_builder.create_batch(train_dataset)
    val_data_X, val_data_Z = batch_builder.create_batch(val_dataset)
    test_data_X, test_data_Z = batch_builder.create_batch(test_dataset)
    
    print(f"  Train graphs: {len(train_data_Z)}")
    print(f"  Val graphs: {len(val_data_Z)}")
    print(f"  Test graphs: {len(test_data_Z)}")
    print(f"  Example graph: {train_data_Z[0].num_nodes} nodes, {train_data_Z[0].edge_index.shape[1]} edges")
    
    return (train_data_X, train_data_Z), (val_data_X, val_data_Z), (test_data_X, test_data_Z)


def train_gnn_decoder(train_data, val_data, args):
    """Train GNN decoder"""
    print("\n" + "="*80)
    print("TRAINING GNN DECODER")
    print("="*80)
    
    train_data_X, train_data_Z = train_data
    val_data_X, val_data_Z = val_data
    
    # Create model
    if args.model_type == 'basic':
        print(f"\nCreating basic GNN decoder...")
        model = GNNDecoder(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_attention=False,
            dropout=args.dropout
        )
    elif args.model_type == 'attention':
        print(f"\nCreating attention GNN decoder...")
        model = GNNDecoder(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_attention=True,
            dropout=args.dropout
        )
    elif args.model_type == 'residual':
        print(f"\nCreating residual GNN decoder...")
        model = ResidualGNNDecoder(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = DecoderLoss(loss_type='bce')
    
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"  Device: {device}")
    
    # Train
    trainer = Trainer(model, optimizer, criterion, device=device, scheduler=scheduler)
    
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(
        train_data_Z,  # Z syndrome detects X errors
        val_data_Z,
        num_epochs=args.epochs,
        verbose=True
    )
    
    # Save model
    if args.save_model:
        save_path = Path(args.output_dir) / f"{args.model_type}_decoder.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'args': vars(args)
        }, save_path)
        print(f"\nModel saved to: {save_path}")
    
    return model, history


def evaluate_all_decoders(code, model, test_data, test_dataset, args):
    """Evaluate GNN and classical decoders"""
    print("\n" + "="*80)
    print("DECODER EVALUATION")
    print("="*80)
    
    test_data_X, test_data_Z = test_data
    
    # Get logical operators if available
    logical_ops_X = None
    if hasattr(code, 'get_logical_operators'):
        logical_ops_X, _ = code.get_logical_operators()
    
    # Create evaluator
    evaluator = DecoderEvaluator(code.H_X, code.H_Z, logical_ops_X)
    
    results = []
    
    # Evaluate GNN
    print("\nEvaluating GNN decoder...")
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    gnn_result = evaluator.evaluate_gnn_decoder(
        model,
        test_data_Z,
        test_dataset['error_X'],
        device=device,
        name=f"GNN ({args.model_type})"
    )
    results.append(gnn_result)
    
    # Evaluate Belief Propagation
    print("Evaluating Belief Propagation decoder...")
    bp_decoder = BeliefPropagationDecoder(code.H_Z, max_iterations=50, damping=0.5)
    bp_result = evaluator.evaluate_classical_decoder(
        bp_decoder,
        test_dataset['syndrome_Z'],
        test_dataset['error_X'],
        name="Belief Propagation"
    )
    results.append(bp_result)
    
    # Evaluate Greedy
    print("Evaluating Greedy decoder...")
    greedy_decoder = GreedyDecoder(code.H_Z, max_iterations=100)
    greedy_result = evaluator.evaluate_classical_decoder(
        greedy_decoder,
        test_dataset['syndrome_Z'],
        test_dataset['error_X'],
        name="Greedy"
    )
    results.append(greedy_result)
    
    # Print comparison
    evaluator.compare_decoders(results)
    
    # Plot results
    if args.plot_results:
        try:
            save_path = Path(args.output_dir) / "decoder_comparison.png" if args.save_plots else None
            evaluator.plot_results(results, save_path=save_path)
        except ImportError:
            print("Matplotlib not available, skipping plots")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='GNN Decoder for Quantum Error Correction')
    
    # Code parameters
    parser.add_argument('--code-type', type=str, default='toric', choices=['toric', 'hypergraph'],
                      help='Type of quantum code')
    parser.add_argument('--code-size', type=int, default=4,
                      help='Code size (L for toric code, n for hypergraph)')
    parser.add_argument('--error-rate', type=float, default=0.05,
                      help='Physical error rate')
    
    # Dataset parameters
    parser.add_argument('--n-train', type=int, default=1000,
                      help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=200,
                      help='Number of validation samples')
    parser.add_argument('--n-test', type=int, default=500,
                      help='Number of test samples')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='basic',
                      choices=['basic', 'attention', 'residual'],
                      help='GNN model architecture')
    parser.add_argument('--hidden-dim', type=int, default=64,
                      help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=5,
                      help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--use-cuda', action='store_true',
                      help='Use CUDA if available')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./outputs',
                      help='Output directory')
    parser.add_argument('--save-model', action='store_true',
                      help='Save trained model')
    parser.add_argument('--plot-results', action='store_true',
                      help='Plot evaluation results')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save plots to file')
    
    args = parser.parse_args()
    
    # Run experiment
    print("\n" + "="*80)
    print("GNN-BASED QUANTUM ERROR CORRECTION")
    print("="*80)
    
    # Setup
    code, simulator = setup_experiment(args)
    
    # Generate data
    datasets = generate_datasets(simulator, args)
    train_dataset, val_dataset, test_dataset = datasets
    
    # Create graphs
    graph_datasets = create_graph_data(code, datasets)
    train_data, val_data, test_data = graph_datasets
    
    # Train model
    model, history = train_gnn_decoder(train_data, val_data, args)
    
    # Evaluate
    results = evaluate_all_decoders(code, model, test_data, test_dataset, args)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  GNN Success Rate: {results[0].success_rate*100:.2f}%")
    print(f"  BP Success Rate: {results[1].success_rate*100:.2f}%")
    print(f"  GNN Speedup: {results[1].avg_decode_time / results[0].avg_decode_time:.1f}x")
    
    # Log experiment
    print("\n" + "="*80)
    print("LOGGING EXPERIMENT")
    print("="*80)
    
    logger = ExperimentLogger()
    
    # Calculate total training time
    training_time = sum(
        trainer.train_losses  # Rough estimate, not exact
    ) if hasattr(trainer, 'train_losses') else 0
    
    # Prepare parameters
    experiment_params = vars(args)
    
    # Log with automatic tracking
    experiment_id = logger.log_experiment(
        params=experiment_params,
        results=results,
        training_time=training_time,
        notes=f"Experiment with {args.model_type} model on L={args.code_size} toric code"
    )
    
    print(f"\n✓ Results logged to experiment_logs/")
    print(f"  View summary: python experiment_logger.py view")
    print(f"  Experiment ID: {experiment_id}")
    
    return results


if __name__ == "__main__":
    # Quick demo without command line args
    import sys
    
    if len(sys.argv) == 1:
        # Run with default arguments
        print("Running with default parameters...")
        print("Use --help to see all options\n")
        
        sys.argv.extend([
            '--code-type', 'toric',
            '--code-size', '4',
            '--error-rate', '0.05',
            '--n-train', '500',
            '--n-val', '100',
            '--n-test', '200',
            '--model-type', 'basic',
            '--hidden-dim', '64',
            '--num-layers', '5',
            '--epochs', '10',
            '--save-model'
        ])
    
    main()
