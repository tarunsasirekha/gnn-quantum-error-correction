"""
Experiment Logger and Results Tracker
Automatically logs all experiments with parameters and results
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class ExperimentLogger:
    """Log and track experiments"""
    
    def __init__(self, log_dir: str = "experiment_logs"):
        """
        Initialize experiment logger
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_file = self.log_dir / "experiments.json"
        self.csv_file = self.log_dir / "experiments.csv"
        self.summary_file = self.log_dir / "SUMMARY.md"
        
        # Initialize files if they don't exist
        if not self.json_file.exists():
            self._init_json()
        if not self.csv_file.exists():
            self._init_csv()
    
    def _init_json(self):
        """Initialize JSON log file"""
        with open(self.json_file, 'w') as f:
            json.dump([], f, indent=2)
    
    def _init_csv(self):
        """Initialize CSV log file"""
        headers = [
            'timestamp', 'experiment_id', 'code_type', 'code_size', 
            'error_rate', 'n_train', 'n_test', 'epochs',
            'model_type', 'hidden_dim', 'num_layers',
            'gnn_success_rate', 'gnn_decode_time', 
            'bp_success_rate', 'bp_decode_time',
            'greedy_success_rate', 'greedy_decode_time',
            'gnn_logical_error', 'training_time', 'notes'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def log_experiment(self, 
                       params: Dict,
                       results: List,
                       training_time: float,
                       notes: str = "") -> str:
        """
        Log a complete experiment
        
        Args:
            params: Dictionary of experiment parameters
            results: List of DecoderResult objects
            training_time: Total training time in seconds
            notes: Optional notes about the experiment
            
        Returns:
            experiment_id: Unique identifier for this experiment
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract results
        gnn_result = results[0] if len(results) > 0 else None
        bp_result = results[1] if len(results) > 1 else None
        greedy_result = results[2] if len(results) > 2 else None
        
        # Create experiment record
        experiment = {
            'timestamp': timestamp,
            'experiment_id': experiment_id,
            'parameters': params,
            'results': {
                'gnn': self._result_to_dict(gnn_result) if gnn_result else {},
                'bp': self._result_to_dict(bp_result) if bp_result else {},
                'greedy': self._result_to_dict(greedy_result) if greedy_result else {}
            },
            'training_time_seconds': training_time,
            'notes': notes
        }
        
        # Save to JSON
        self._append_json(experiment)
        
        # Save to CSV
        self._append_csv(experiment)
        
        # Update summary
        self._update_summary()
        
        print(f"\n✓ Experiment logged: {experiment_id}")
        print(f"  Results saved to: {self.log_dir}")
        
        return experiment_id
    
    def _result_to_dict(self, result) -> Dict:
        """Convert DecoderResult to dictionary"""
        if result is None:
            return {}
        
        return {
            'name': result.name,
            'success_rate': result.success_rate,
            'logical_error_rate': result.logical_error_rate,
            'avg_decode_time': result.avg_decode_time,
            'avg_error_weight': result.avg_error_weight,
            'avg_residual_weight': result.avg_residual_weight,
            'frame_error_rate': result.frame_error_rate
        }
    
    def _append_json(self, experiment: Dict):
        """Append experiment to JSON log"""
        # Read existing
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        # Append new
        data.append(experiment)
        
        # Write back
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _append_csv(self, experiment: Dict):
        """Append experiment to CSV log"""
        params = experiment['parameters']
        gnn = experiment['results'].get('gnn', {})
        bp = experiment['results'].get('bp', {})
        greedy = experiment['results'].get('greedy', {})
        
        row = {
            'timestamp': experiment['timestamp'],
            'experiment_id': experiment['experiment_id'],
            'code_type': params.get('code_type', 'toric'),
            'code_size': params.get('code_size', 0),
            'error_rate': params.get('error_rate', 0.0),
            'n_train': params.get('n_train', 0),
            'n_test': params.get('n_test', 0),
            'epochs': params.get('epochs', 0),
            'model_type': params.get('model_type', 'basic'),
            'hidden_dim': params.get('hidden_dim', 0),
            'num_layers': params.get('num_layers', 0),
            'gnn_success_rate': gnn.get('success_rate', 0.0),
            'gnn_decode_time': gnn.get('avg_decode_time', 0.0),
            'bp_success_rate': bp.get('success_rate', 0.0),
            'bp_decode_time': bp.get('avg_decode_time', 0.0),
            'greedy_success_rate': greedy.get('success_rate', 0.0),
            'greedy_decode_time': greedy.get('avg_decode_time', 0.0),
            'gnn_logical_error': gnn.get('logical_error_rate', 0.0),
            'training_time': experiment['training_time_seconds'],
            'notes': experiment['notes']
        }
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def _update_summary(self):
        """Generate markdown summary of all experiments"""
        # Read all experiments
        with open(self.json_file, 'r') as f:
            experiments = json.load(f)
        
        if not experiments:
            return
        
        # Generate summary
        summary = "# Experiment Results Summary\n\n"
        summary += f"**Total Experiments:** {len(experiments)}\n\n"
        summary += f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Table of all experiments
        summary += "## All Experiments\n\n"
        summary += "| ID | Date | Code | Arch | Error% | Train | Epochs | GNN% | BP% | Time |\n"
        summary += "|----|------|------|------|--------|-------|--------|------|-----|------|\n"
        
        for exp in experiments[-20:]:  # Last 20 experiments
            params = exp['parameters']
            gnn = exp['results'].get('gnn', {})
            bp = exp['results'].get('bp', {})
            
            exp_id = exp['experiment_id'][-8:]  # Last 8 chars
            date = exp['timestamp'][:10]
            code = f"L={params.get('code_size', '?')}"
            arch = params.get('model_type', 'basic')
            error = f"{params.get('error_rate', 0)*100:.1f}%"
            train = params.get('n_train', 0)
            epochs = params.get('epochs', 0)
            gnn_succ = f"{gnn.get('success_rate', 0)*100:.1f}%"
            bp_succ = f"{bp.get('success_rate', 0)*100:.1f}%"
            time_min = f"{exp['training_time_seconds']/60:.1f}m"
            
            summary += f"| {exp_id} | {date} | {code} | {arch} | {error} | {train} | {epochs} | {gnn_succ} | {bp_succ} | {time_min} |\n"
        
        # Best results
        summary += "\n## Best Results\n\n"
        
        # Find best GNN success rate
        best_gnn = max(experiments, 
                      key=lambda x: x['results'].get('gnn', {}).get('success_rate', 0))
        
        summary += "### Highest GNN Success Rate\n"
        summary += f"- **Experiment ID:** {best_gnn['experiment_id']}\n"
        summary += f"- **Success Rate:** {best_gnn['results']['gnn']['success_rate']*100:.2f}%\n"
        summary += f"- **Code Size:** L={best_gnn['parameters']['code_size']}\n"
        summary += f"- **Architecture:** {best_gnn['parameters']['model_type']}\n"
        summary += f"- **Error Rate:** {best_gnn['parameters']['error_rate']*100:.1f}%\n\n"
        
        # Architecture comparison
        summary += "## Architecture Comparison\n\n"
        
        arch_stats = {}
        for exp in experiments:
            arch = exp['parameters'].get('model_type', 'basic')
            gnn_succ = exp['results'].get('gnn', {}).get('success_rate', 0)
            
            if arch not in arch_stats:
                arch_stats[arch] = []
            arch_stats[arch].append(gnn_succ)
        
        summary += "| Architecture | Avg Success | Best | Count |\n"
        summary += "|--------------|-------------|------|-------|\n"
        
        for arch, successes in arch_stats.items():
            avg = sum(successes) / len(successes) * 100
            best = max(successes) * 100
            count = len(successes)
            summary += f"| {arch} | {avg:.2f}% | {best:.2f}% | {count} |\n"
        
        # Code size scaling
        summary += "\n## Code Size Scaling\n\n"
        
        size_stats = {}
        for exp in experiments:
            size = exp['parameters'].get('code_size', 0)
            gnn_succ = exp['results'].get('gnn', {}).get('success_rate', 0)
            
            if size not in size_stats:
                size_stats[size] = []
            size_stats[size].append(gnn_succ)
        
        summary += "| Code Size (L) | Qubits | Avg Success | Count |\n"
        summary += "|---------------|--------|-------------|-------|\n"
        
        for size in sorted(size_stats.keys()):
            qubits = 2 * size * size
            successes = size_stats[size]
            avg = sum(successes) / len(successes) * 100
            count = len(successes)
            summary += f"| {size} | {qubits} | {avg:.2f}% | {count} |\n"
        
        # Write summary
        with open(self.summary_file, 'w') as f:
            f.write(summary)
    
    def get_experiments(self, 
                       code_size: Optional[int] = None,
                       model_type: Optional[str] = None,
                       error_rate: Optional[float] = None) -> List[Dict]:
        """
        Get experiments matching filters
        
        Args:
            code_size: Filter by code size
            model_type: Filter by architecture
            error_rate: Filter by error rate
            
        Returns:
            List of matching experiments
        """
        with open(self.json_file, 'r') as f:
            experiments = json.load(f)
        
        filtered = experiments
        
        if code_size is not None:
            filtered = [e for e in filtered 
                       if e['parameters'].get('code_size') == code_size]
        
        if model_type is not None:
            filtered = [e for e in filtered 
                       if e['parameters'].get('model_type') == model_type]
        
        if error_rate is not None:
            filtered = [e for e in filtered 
                       if abs(e['parameters'].get('error_rate', 0) - error_rate) < 0.001]
        
        return filtered
    
    def print_summary(self):
        """Print summary to console"""
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                print(f.read())
        else:
            print("No experiments logged yet.")


def view_results():
    """View all logged results"""
    logger = ExperimentLogger()
    logger.print_summary()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "view":
        view_results()
    else:
        print("Experiment Logger")
        print("=" * 60)
        print("\nUsage:")
        print("  python experiment_logger.py view     # View all results")
        print("\nOr import in your code:")
        print("  from experiment_logger import ExperimentLogger")
        print("  logger = ExperimentLogger()")
        print("  logger.log_experiment(params, results, training_time)")
