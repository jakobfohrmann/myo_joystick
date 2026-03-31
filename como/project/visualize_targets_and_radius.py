"""Visualize fixed targets with radius and joystick positions"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import fixed targets from curriculum callback
from models.callbacks import CurriculumCallback

# Import environment class to get target radius constant
from envs.thumb_reach import ThumbReachEnvV0


def load_joystick_data(log_dir):
    """Load joystick position data from log directory"""
    all_positions_file = os.path.join(log_dir, "joystick_all_positions.npy")
    final_positions_file = os.path.join(log_dir, "joystick_episode_final_positions.npy")
    episode_targets_file = os.path.join(log_dir, "target_episode_positions.npy")
    
    all_positions = None
    final_positions = None
    episode_targets = None
    
    if os.path.exists(all_positions_file):
        all_positions = np.load(all_positions_file)
        print(f"✓ Loaded {len(all_positions)} total joystick positions")
    
    if os.path.exists(final_positions_file):
        final_positions = np.load(final_positions_file)
        print(f"✓ Loaded {len(final_positions)} episode final positions")
    
    if os.path.exists(episode_targets_file):
        episode_targets = np.load(episode_targets_file)
        print(f"✓ Loaded {len(episode_targets)} episode target positions")
    
    return all_positions, final_positions, episode_targets


def visualize_targets_with_radius(
    all_positions=None,
    final_positions=None,
    episode_targets=None,
    target_radius=0.5,
    save_path=None,
    show_plot=True
):
    """
    Visualize fixed targets with radius circles and joystick positions.
    
    Args:
        all_positions: Array of shape (N, 2) with all joystick positions (optional)
        final_positions: Array of shape (M, 2) with final positions per episode (optional)
        episode_targets: Array of shape (M, 2) with target positions per episode (optional)
        target_radius: Radius of the target (default: 0.5 = 50%)
        save_path: Path to save the figure
        show_plot: Whether to display the plot
    """
    # Get fixed targets from curriculum (direkt aus dem Code)
    fixed_targets = CurriculumCallback.FIXED_TARGETS
    print(f"Using fixed targets from code: {[list(t) for t in fixed_targets]}")
    target_names = [f"Target {i+1}" for i in range(len(fixed_targets))]
    colors = ['red', 'blue', 'green', 'orange'] + ['gray'] * max(0, len(fixed_targets) - 4)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    for i, (target, name, color) in enumerate(zip(fixed_targets, target_names, colors)):
        ax.scatter(target[0], target[1], c=color, s=300, marker='*', 
                  edgecolors='black', linewidths=2, zorder=10, label=f'{name}: {target}')
        circle = Circle(target, target_radius, fill=True, alpha=0.2, 
                       color=color, edgecolor=color, linewidth=2, linestyle='--', zorder=5)
        ax.add_patch(circle)
        ax.text(target[0], target[1] + 0.15, f'T{i+1}', 
               fontsize=12, fontweight='bold', ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot joystick positions if available
    if all_positions is not None and len(all_positions) > 0:
        # Sample positions if too many (for performance)
        if len(all_positions) > 10000:
            indices = np.random.choice(len(all_positions), 10000, replace=False)
            sampled_positions = all_positions[indices]
        else:
            sampled_positions = all_positions
        
        ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1], 
                  c='gray', s=1, alpha=0.3, label=f'Joystick Positions (n={len(sampled_positions):,})', zorder=1)
    
    # Plot final positions per episode if available
    if final_positions is not None and len(final_positions) > 0:
        ax.scatter(final_positions[:, 0], final_positions[:, 1], 
                  c='cyan', s=50, edgecolors='black', linewidths=1, 
                  marker='o', alpha=0.6, 
                  label=f'Episode Final Positions (n={len(final_positions)})', zorder=8)
    
    # Plot episode targets if available (should match fixed targets)
    if episode_targets is not None and len(episode_targets) > 0:
        # Count how many times each target was used
        unique_targets, counts = np.unique(episode_targets, axis=0, return_counts=True)
        for target, count in zip(unique_targets, counts):
            ax.scatter(target[0], target[1], c='yellow', s=200, 
                      edgecolors='black', linewidths=2, marker='s', alpha=0.7,
                      zorder=9)
            ax.text(target[0], target[1] - 0.15, f'{count}x', 
                   fontsize=10, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Add center lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)
    
    # Set labels and title
    ax.set_xlabel('Joystick X Position (normalized [-1, 1])', fontsize=14)
    ax.set_ylabel('Joystick Y Position (normalized [-1, 1])', fontsize=14)
    ax.set_title(f'Fixed Targets mit Radius (Radius = {target_radius} = {target_radius*100:.0f}%)', 
                fontsize=16, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add info text box
    info_text = f"Target Radius: {target_radius} ({target_radius*100:.0f}%)\n"
    info_text += f"Absoluter Radius: {target_radius * 0.35:.3f} Radians\n"
    info_text += f"Maximaler Joystick-Range: ±0.35 Radians\n"
    info_text += f"Normalisierter Range: [-1, 1]"
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Target labels (nur wenn genau 4 Targets wie in TARGETS_4)
    if len(fixed_targets) == 4:
        ax.text(0.6, 0.6, 'T1', fontsize=16, fontweight='bold', 
               color='red', alpha=0.3, ha='center', va='center')
        ax.text(-0.6, 0.6, 'T2', fontsize=16, fontweight='bold', 
               color='blue', alpha=0.3, ha='center', va='center')
        ax.text(-0.6, -0.6, 'T3', fontsize=16, fontweight='bold', 
               color='green', alpha=0.3, ha='center', va='center')
        ax.text(0.6, -0.6, 'T4', fontsize=16, fontweight='bold', 
               color='orange', alpha=0.3, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization to {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize fixed targets with radius and joystick positions"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Path to log directory containing joystick position data (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the visualization (default: target_radius_visualization.png)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Target radius (default: extracted from environment code = 0.5 = 50%%)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save)"
    )
    
    args = parser.parse_args()
    
    # Get target radius from environment code if not specified
    if args.radius is None:
        print("Extracting target radius from environment code...")
        args.radius = ThumbReachEnvV0.TARGET_RADIUS  # Direkt aus der Klassen-Konstante
        print(f"✓ Using target radius from code: {args.radius} ({args.radius*100:.0f}%)")
    
    # Load joystick data if log_dir provided
    all_positions = None
    final_positions = None
    episode_targets = None
    
    if args.log_dir:
        print("Loading joystick position data...")
        all_positions, final_positions, episode_targets = load_joystick_data(args.log_dir)
    else:
        print("No log directory provided - showing only targets without joystick data")
    
    # Determine output path
    if args.output is None:
        if args.log_dir:
            args.output = os.path.join(args.log_dir, "target_radius_visualization.png")
        else:
            args.output = "target_radius_visualization.png"
    
    # Create visualization
    print("Creating visualization...")
    print(f"Using fixed targets from CurriculumCallback: {[list(t) for t in CurriculumCallback.FIXED_TARGETS]}")
    visualize_targets_with_radius(
        all_positions=all_positions,
        final_positions=final_positions,
        episode_targets=episode_targets,
        target_radius=args.radius,
        save_path=args.output,
        show_plot=not args.no_show
    )
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
