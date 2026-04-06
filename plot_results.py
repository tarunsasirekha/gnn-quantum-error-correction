"""
Visualization Script for GNN Quantum Error Correction Results
Generates 5 publication-quality plots from experiment logs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_experiment_data(csv_path='experiment_logs/experiments.csv'):
    """Load experiment data from CSV"""
    df = pd.read_csv(csv_path)
    return df

def plot_1_gnn_vs_classical(df, save_path='plots/1_gnn_vs_classical.png'):
    """Plot 1: GNN vs Classical Decoders Success Rate"""
    
    # Get average performance for each decoder
    avg_gnn = df['gnn_success_rate'].mean() * 100
    avg_bp = df['bp_success_rate'].mean() * 100
    avg_greedy = df['greedy_success_rate'].mean() * 100
    
    # Also get best performance
    best_gnn = df['gnn_success_rate'].max() * 100
    best_bp = df['bp_success_rate'].max() * 100
    best_greedy = df['greedy_success_rate'].max() * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    decoders = ['GNN\n(Best)', 'BP\n(Best)', 'Greedy\n(Best)', 
                'GNN\n(Avg)', 'BP\n(Avg)', 'Greedy\n(Avg)']
    success_rates = [best_gnn, best_bp, best_greedy, avg_gnn, avg_bp, avg_greedy]
    colors = ['#2E86AB', '#A23B72', '#F18F01', 
              '#2E86AB', '#A23B72', '#F18F01']
    alphas = [1.0, 1.0, 1.0, 0.6, 0.6, 0.6]
    
    bars = ax.bar(decoders, success_rates, color=colors, edgecolor='black', linewidth=1.5)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Decoder Performance Comparison\nGNN vs Classical Methods', 
                 fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 90%
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend()
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_2_decode_time_comparison(df, save_path='plots/2_decode_time.png'):
    """Plot 2: Decode Time Comparison (Log Scale)"""
    
    # Get average decode times
    avg_gnn_time = df['gnn_decode_time'].mean() * 1000  # Convert to ms
    avg_bp_time = df['bp_decode_time'].mean() * 1000
    avg_greedy_time = df['greedy_decode_time'].mean() * 1000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    decoders = ['GNN', 'Belief\nPropagation', 'Greedy']
    times = [avg_gnn_time, avg_bp_time, avg_greedy_time]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax.bar(decoders, times, color=colors, edgecolor='black', linewidth=1.5)
    
    
    # Add value labels
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} ms',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Calculate speedup
    speedup = avg_bp_time / avg_gnn_time
    ax.text(0.5, 0.95, f'GNN is {speedup:.1f}× faster than BP!', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Decode Time (milliseconds)', fontweight='bold')
    ax.set_title('Decoding Latency Comparison\nLower is Better', 
                 fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_3_architecture_comparison(df, save_path='plots/3_architecture_comparison.png'):
    """Plot 3: GNN Architecture Comparison"""
    
    # Group by architecture
    arch_stats = df.groupby('model_type').agg({
        'gnn_success_rate': ['mean', 'std', 'max'],
        'gnn_decode_time': 'mean'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    architectures = arch_stats['model_type'].values
    success_mean = arch_stats[('gnn_success_rate', 'mean')].values * 100
    success_std = arch_stats[('gnn_success_rate', 'std')].values * 100
    success_max = arch_stats[('gnn_success_rate', 'max')].values * 100
    decode_time = arch_stats[('gnn_decode_time', 'mean')].values * 1000
    
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(architectures)]
    
    # Plot 1: Success Rate
    x_pos = np.arange(len(architectures))
    bars1 = ax1.bar(x_pos - 0.2, success_mean, 0.4, 
                    label='Average', color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + 0.2, success_max, 0.4,
                    label='Best', color=colors, alpha=1.0,
                    edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax1.errorbar(x_pos - 0.2, success_mean, yerr=success_std, 
                 fmt='none', color='black', capsize=5, alpha=0.5)
    
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Architecture Performance', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(architectures, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, success_mean)):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Decode Time
    bars3 = ax2.bar(architectures, decode_time, color=colors,
                    edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars3, decode_time):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Decode Time (ms)', fontweight='bold')
    ax2.set_title('Architecture Speed', fontweight='bold')
    ax2.set_xticklabels(architectures, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_4_code_size_scaling(df, save_path='plots/4_code_size_scaling.png'):
    """Plot 4: Performance vs Code Size (Scalability)"""
    
    # Group by code size
    size_stats = df.groupby('code_size').agg({
        'gnn_success_rate': 'mean',
        'bp_success_rate': 'mean',
        'gnn_decode_time': 'mean',
        'bp_decode_time': 'mean'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    code_sizes = size_stats['code_size'].values
    n_qubits = 2 * code_sizes**2  # For toric codes
    
    # Plot 1: Success Rate vs Code Size
    ax1.plot(code_sizes, size_stats['gnn_success_rate']*100, 
             'o-', color='#2E86AB', linewidth=2.5, markersize=10,
             label='GNN', markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(code_sizes, size_stats['bp_success_rate']*100,
             's-', color='#A23B72', linewidth=2.5, markersize=10,
             label='BP', markeredgecolor='black', markeredgewidth=1.5)
    
    ax1.set_xlabel('Code Size (L)', fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Scalability: Success Rate vs Code Size', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(70, 100)
    
    # Add qubit count on secondary x-axis
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(code_sizes)
    ax1_top.set_xticklabels([f'{n}q' for n in n_qubits], fontsize=9)
    ax1_top.set_xlabel('Number of Qubits', fontsize=10, style='italic')
    
    # Plot 2: Decode Time vs Code Size
    ax2.plot(code_sizes, size_stats['gnn_decode_time']*1000,
             'o-', color='#2E86AB', linewidth=2.5, markersize=10,
             label='GNN', markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(code_sizes, size_stats['bp_decode_time']*1000,
             's-', color='#A23B72', linewidth=2.5, markersize=10,
             label='BP', markeredgecolor='black', markeredgewidth=1.5)
    
    ax2.set_xlabel('Code Size (L)', fontweight='bold')
    ax2.set_ylabel('Decode Time (ms)', fontweight='bold')
    ax2.set_title('Scalability: Decode Time vs Code Size', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_5_speed_accuracy_tradeoff(df, save_path='plots/5_speed_accuracy_tradeoff.png'):
    """Plot 5: Speed vs Accuracy Trade-off (Scatter)"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # GNN points
    gnn_success = df['gnn_success_rate'].values * 100
    gnn_time = df['gnn_decode_time'].values * 1000
    
    # BP points
    bp_success = df['bp_success_rate'].values * 100
    bp_time = df['bp_decode_time'].values * 1000
    
    # Greedy points
    greedy_success = df['greedy_success_rate'].values * 100
    greedy_time = df['greedy_decode_time'].values * 1000
    
    # Plot with different markers for each decoder
    ax.scatter(gnn_time, gnn_success, s=200, alpha=0.7, 
              c='#2E86AB', marker='o', edgecolors='black', linewidths=2,
              label='GNN', zorder=3)
    ax.scatter(bp_time, bp_success, s=200, alpha=0.7,
              c='#A23B72', marker='s', edgecolors='black', linewidths=2,
              label='BP', zorder=3)
    ax.scatter(greedy_time, greedy_success, s=200, alpha=0.7,
              c='#F18F01', marker='^', edgecolors='black', linewidths=2,
              label='Greedy', zorder=3)
    
    # Add arrows and annotations
    avg_gnn_time = gnn_time.mean()
    avg_gnn_success = gnn_success.mean()
    avg_bp_time = bp_time.mean()
    avg_bp_success = bp_success.mean()
    
    # Arrow from BP to GNN
    ax.annotate('', xy=(avg_gnn_time, avg_gnn_success), 
                xytext=(avg_bp_time, avg_bp_success),
                arrowprops=dict(arrowstyle='->', lw=3, color='green', alpha=0.6))
    
    speedup = avg_bp_time / avg_gnn_time
    accuracy_loss = avg_bp_success - avg_gnn_success
    
    ax.text((avg_gnn_time + avg_bp_time)/2, (avg_gnn_success + avg_bp_success)/2 + 2,
            f'{speedup:.1f}× faster\n{accuracy_loss:.1f}% accuracy loss',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlabel('Decode Time (milliseconds) - Log Scale', fontweight='bold', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Speed-Accuracy Trade-off\nIdeal: Top-Left Corner (Fast & Accurate)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xscale('log')
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(70, 100)
    
    # Add "Better" annotations
    ax.text(0.02, 0.98, '← Faster', transform=ax.transAxes,
            fontsize=11, va='top', style='italic', color='green')
    ax.text(0.98, 0.02, 'More Accurate ↑', transform=ax.transAxes,
            fontsize=11, ha='right', style='italic', color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def generate_all_plots(csv_path='experiment_logs/experiments.csv'):
    """Generate all 5 plots"""
    print("="*60)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*60)
    
    # Load data
    print("\nLoading experiment data...")
    df = load_experiment_data(csv_path)
    print(f"✓ Loaded {len(df)} experiments")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_1_gnn_vs_classical(df)
    plot_2_decode_time_comparison(df)
    plot_3_architecture_comparison(df)
    plot_4_code_size_scaling(df)
    plot_5_speed_accuracy_tradeoff(df)
    
    print("\n" + "="*60)
    print("✓ ALL PLOTS GENERATED!")
    print("="*60)
    print("\nPlots saved in: plots/")
    print("  1. GNN vs Classical Decoders")
    print("  2. Decode Time Comparison")
    print("  3. Architecture Comparison")
    print("  4. Code Size Scaling")
    print("  5. Speed-Accuracy Trade-off")
    print("\nUse these in your presentation, report, or LinkedIn post!")

if __name__ == "__main__":
    generate_all_plots()
