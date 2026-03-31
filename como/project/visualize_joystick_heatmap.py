"""Visualize joystick movement heatmap with episode final positions as splash points"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from matplotlib.patches import Circle

# Episode final/target splash points:
SPLASH_EPISODES_MAX = None  # None => splash points for all episodes
# SPLASH_EPISODES_MAX = 2000  # uncomment => splash points for last N episodes
TARGET_RADII_DEFAULT = [0.3, 0.2, 0.1, 0.05]
SKIP_FIRST_EPISODES = 10
# Paper: ausblenden der Statistik-Box oben links (Joystick counts, Episodes, …).
# Auf True setzen, um sie wieder anzuzeigen.
SHOW_STATS_TEXT_BOX = False
# Paper: Legende oben rechts (Final/Target Splash-Punkte). Auf True setzen zum Wiederanzeigen.
SHOW_SPLASH_LEGEND = False
# Feste Curriculum-Targets: einheitliche Darstellung (keine rot/blau/grün/orange Sterne/Kreise)
FIXED_TARGET_MARKER_COLOR = "0.2"
FIXED_TARGET_RADIUS_LINE_COLOR = "0.35"


def _splash_slice(arr, max_episodes=SPLASH_EPISODES_MAX):
    """Return splash-point rows (last N if set, otherwise all)."""
    if arr is None or len(arr) == 0:
        return None
    if max_episodes is None:
        return np.asarray(arr)
    return np.asarray(arr)[-max_episodes:]


def _last_n_rows(arr, last_n):
    """Return all rows or only last N rows."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if last_n is None:
        return arr
    if last_n <= 0:
        return arr[:0]
    return arr[-last_n:]


def _skip_first_rows(arr, skip_n):
    """Skip the first N rows from an array-like object."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if skip_n is None or skip_n <= 0:
        return arr
    return arr[skip_n:]


def ensure_dir_for(path):
    """Create parent directory of path if it does not exist."""
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_joystick_data(log_dir):
    """Load joystick position and target data from log directory"""
    all_positions_file = os.path.join(log_dir, "joystick_all_positions.npy")
    final_positions_file = os.path.join(log_dir, "joystick_episode_final_positions.npy")
    all_targets_file = os.path.join(log_dir, "target_all_positions.npy")
    episode_targets_file = os.path.join(log_dir, "target_episode_positions.npy")
    metadata_file = os.path.join(log_dir, "joystick_episode_metadata.npz")
    
    all_positions = None
    final_positions = None
    all_targets = None
    episode_targets = None
    metadata = None
    
    if os.path.exists(all_positions_file):
        all_positions = np.load(all_positions_file)
        print(f"✓ Loaded {len(all_positions)} total joystick positions")
    else:
        print(f"⚠️  Warning: {all_positions_file} not found")
    
    if os.path.exists(final_positions_file):
        final_positions = np.load(final_positions_file)
        print(f"✓ Loaded {len(final_positions)} episode final positions")
    else:
        print(f"⚠️  Warning: {final_positions_file} not found")
    
    if os.path.exists(all_targets_file):
        all_targets = np.load(all_targets_file)
        print(f"✓ Loaded {len(all_targets)} total target positions")
    else:
        print(f"  Warning: {all_targets_file} not found")
    
    if os.path.exists(episode_targets_file):
        episode_targets = np.load(episode_targets_file)
        print(f" Loaded {len(episode_targets)} episode target positions")
    else:
        print(f"  Warning: {episode_targets_file} not found")
    
    if os.path.exists(metadata_file):
        metadata = np.load(metadata_file)
        print(f" Loaded episode metadata")
    
    return all_positions, final_positions, all_targets, episode_targets, metadata


def log_dir_title(log_dir: str, kde: bool = False, targets_only: bool = False) -> str:
    """Figure title from log directory (last path segment)."""
    base = os.path.basename(os.path.normpath(os.path.abspath(log_dir)))
    if not base:
        base = log_dir or "heatmap"
    parts = [base]
    if targets_only:
        parts.append("targets")
    if kde:
        parts.append("KDE")
    return " — ".join(parts)


def add_timestamp_to_filename(filepath):
    """Add timestamp to filename before the extension"""
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{name}_{timestamp}{ext}"
    return os.path.join(directory, new_filename)


def create_heatmap(all_positions, final_positions=None, all_targets=None, episode_targets=None,
                   bins=50, save_path=None, title="Joystick heatmap", show_plot=True,
                   show_target_radius=True, target_radius=None, splash_max=SPLASH_EPISODES_MAX):
    """
    Create a heatmap of joystick positions with splash points for episode final positions and targets.
    
    Args:
        all_positions: Array of shape (N, 2) with all joystick positions
        final_positions: Array of shape (M, 2) with final positions per episode
        all_targets: Array of shape (K, 2) with all target positions (optional, for target heatmap overlay)
        episode_targets: Array of shape (M, 2) with target position per episode (for splash points)
        bins: Number of bins for heatmap
        save_path: Path to save the figure
        title: Title for the plot
        show_plot: Whether to display the plot
    """
    if all_positions is None or len(all_positions) == 0:
        print("  No position data available for heatmap")
        return
    
    # Extract x and y coordinates
    x = all_positions[:, 0]
    y = all_positions[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create 2D histogram for heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-1, 1], [-1, 1]])
    
    # Apply log scale for better visualization (avoid log(0))
    heatmap_log = np.log1p(heatmap)
    
    # Create custom colormap (blue to red, with transparency)
    colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000', '#800000']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap_log.T, origin='lower', extent=extent, 
                   cmap=cmap, aspect='auto', alpha=0.7, interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log Density (log(1 + count))', rotation=270, labelpad=20)
    
    # Add splash points for episode final positions (last N if SPLASH_EPISODES_MAX is set)
    legend_handles = []
    splash_final = _splash_slice(final_positions, splash_max)
    if splash_final is not None:
        n_tot = len(final_positions)
        final_x = splash_final[:, 0]
        final_y = splash_final[:, 1]
        lbl = (
            f'Joystick Final Positions (last {len(splash_final)} of {n_tot})'
            if n_tot > len(splash_final)
            else f'Joystick Final Positions (n={n_tot})'
        )
        scatter1 = ax.scatter(final_x, final_y, c='white', s=100, edgecolors='black', 
                  linewidths=2, marker='o', alpha=0.8, label=lbl,
                  zorder=10)
        legend_handles.append(scatter1)
    
    # Add target positions (if available)
    splash_targets = _splash_slice(episode_targets, splash_max)
    if splash_targets is not None:
        n_tt = len(episode_targets)
        target_x = splash_targets[:, 0]
        target_y = splash_targets[:, 1]
        lbl_t = (
            f'Target Positions (last {len(splash_targets)} of {n_tt})'
            if n_tt > len(splash_targets)
            else f'Target Positions (n={n_tt})'
        )
        scatter2 = ax.scatter(target_x, target_y, c='lime', s=150, edgecolors='darkgreen', 
                  linewidths=2, marker='*', alpha=0.9, label=lbl_t,
                  zorder=11)
        legend_handles.append(scatter2)
    
    # Add fixed targets with radius circles if enabled
    if show_target_radius:
        from models.callbacks import CurriculumCallback
        fixed_targets = CurriculumCallback.FIXED_TARGETS
        radii = TARGET_RADII_DEFAULT
        mc = FIXED_TARGET_MARKER_COLOR
        rc = FIXED_TARGET_RADIUS_LINE_COLOR

        for i, target in enumerate(fixed_targets):
            ax.scatter(
                target[0],
                target[1],
                c=mc,
                s=200,
                marker="*",
                edgecolors="black",
                linewidths=2,
                zorder=12,
                alpha=0.95,
            )
            for r in radii:
                circle = Circle(
                    target,
                    r,
                    fill=False,
                    color=rc,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.75,
                    zorder=6,
                )
                ax.add_patch(circle)
            ax.text(target[0], target[1] + 0.12, f'T{i+1}', 
                   fontsize=10, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1.0), zorder=25)
    
    if SHOW_SPLASH_LEGEND and legend_handles:
        legend = ax.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=1.0)
        legend.set_zorder(30)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Joystick X Position (normalized)', fontsize=12)
    ax.set_ylabel('Joystick Y Position (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Add center lines
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    if SHOW_STATS_TEXT_BOX:
        stats_text = f"Joystick Positions: {len(all_positions):,}\n"
        if final_positions is not None and len(final_positions) > 0:
            stats_text += f"Episodes: {len(final_positions):,}\n"
        if all_targets is not None and len(all_targets) > 0:
            stats_text += f"Target Samples: {len(all_targets):,}"
        if show_target_radius:
            stats_text += "\nTarget Radii: 0.30, 0.20, 0.10, 0.05"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=1.0),
            zorder=30,
        )
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved heatmap to {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_target_single_curriculum(save_path=None, title="Target (Curriculum)", show_plot=True):
    """
    Ein Target bei (0, 0) mit drei Radius-Kreisen (Viridis-Farben), Kern in der Mitte, Pfeil nach innen.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cx, cy = 0.0, 0.0
    # Drei Radien (äußerer, mittlerer, innerer) – gleiche Farblogik wie bei create_targets_blank
    radii_three = [0.35, 0.55, 0.75]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(radii_three)))[::-1]
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [-1, 1]")
    ax.set_ylabel("y [-1, 1]")
    # Drei Radius-Kreise (gleiche Viridis-Farben wie bei create_targets_blank: klein → groß)
    for r, color in zip(radii_three, colors):
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=color, linewidth=2.0, alpha=0.9))
    # Kern (Punkt) in der Mitte
    r_core = 0.04
    ax.add_patch(Circle((cx, cy), r_core, fill=True, facecolor="black", edgecolor="none", zorder=5))
    r_big = radii_three[-1]  # für Pfeil/Text
    # Pfeil von außen nach innen
    ax.annotate("", xy=(0, -0.2), xytext=(0, -r_big - 0.15),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2.5))
    ax.text(0, -r_big - 0.28, "Radius decreases (Curriculum)",
            fontsize=11, ha="center", va="top", color="gray",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_targets_blank(targets, save_path=None, title="Targets", in_01=False, show_plot=True,
                        radii=None, quadrant_labels=None, success_rates=None, show_radius_curriculum=True):
    """
    Erstellt ein leeres Plot nur mit den Target-Positionen (kein Heatmap).
    radii: Liste von Radien zum Einzeichnen (z.B. [0.1, 0.2, 0.3]). Im [0,1]-Raum werden sie halbiert.
    quadrant_labels: Optionale Beschriftung pro Target (z.B. ["Q1", "Q2", "Q3", "Q4"]).
    success_rates: Optionale Liste/Array der Länge n_targets (0..1); färbt Kreise rot (niedrig) bis grün (hoch).
    show_radius_curriculum: Wenn True, zeige Veranschaulichung „Radius wird kleiner“ (Curriculum) mit Pfeil.
    """
    if targets is None or len(targets) == 0:
        print("  No target positions for blank plot.")
        return
    targets = np.asarray(targets)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(8, 8))
    if in_01:
        # [0,1]-Raum (wie im Env nach Mapping): Radii halbieren
        x, y = targets[:, 0], targets[:, 1]
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("x [0,1]")
        ax.set_ylabel("y [0,1]")
        draw_radii = [r / 2.0 for r in radii] if radii else None
    else:
        # [-1,1]-Raum (TARGETS_20)
        x, y = targets[:, 0], targets[:, 1]
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [-1,1]")
        ax.set_ylabel("y [-1,1]")
        draw_radii = radii
    # Kreise für jeden Radius um jedes Target
    from matplotlib.lines import Line2D
    legend_handles = []
    if draw_radii and len(draw_radii) > 0:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(draw_radii)))[::-1]
        for cx, cy in targets:
            for r, color in zip(draw_radii, colors):
                ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=color, linewidth=1.2, alpha=0.8))
        for i, r in enumerate(draw_radii):
            label = f"r = {radii[i]}" if radii else f"r = {r:.2f}"
            legend_handles.append(Line2D([0], [0], color=colors[i], linewidth=2, label=label))
    if success_rates is not None and len(success_rates) == len(targets):
        success_rates = np.asarray(success_rates, dtype=float)
        # Größe: klein (niedrig) bis groß (hoch), wie bei Target-Sampling-Veranschaulichung
        sizes = 50 + 180 * success_rates
        sc = ax.scatter(x, y, c=success_rates, s=sizes, cmap="RdYlGn", vmin=0, vmax=1,
                        edgecolors="black", linewidths=1.5, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, label="Success rate")
    else:
        ax.scatter(x, y, c="tab:orange", s=120, edgecolors="black", linewidths=1.5, zorder=5)
    if quadrant_labels is not None and len(quadrant_labels) == len(targets):
        offset = 0.12 if not in_01 else 0.06
        for (cx, cy), label in zip(targets, quadrant_labels):
            ax.text(cx, cy + offset, label, fontsize=9, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    if success_rates is None or len(success_rates) != len(targets):
        legend_handles.insert(0, Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange",
                                       markeredgecolor="black", markersize=10, label="Targets"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    # Veranschaulichung „Radius wird kleiner“: ein großes Ziel bei (0, 0) + Pfeil
    if show_radius_curriculum:
        cx0, cy0 = 0.0, 0.0
        r_big = 0.18 if not in_01 else 0.12
        # Ein großes Target bei (0, 0)
        ax.add_patch(Circle((cx0, cy0), r_big, fill=False, edgecolor="gray", linewidth=2, alpha=0.9))
        # Pfeil am Target nach unten: Radius wird kleiner
        ax.annotate("", xy=(cx0, cy0 - r_big - 0.08), xytext=(cx0, cy0 - r_big + 0.02),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=2))
        ax.text(cx0, cy0 - r_big - 0.14, "Radius decreases\n(Curriculum)",
                fontsize=9, ha="center", va="top", color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} (n={len(targets)})" + (f", Radien {radii}" if radii else ""))
    plt.tight_layout()
    if save_path:
        ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved blank targets plot to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_kde_heatmap(all_positions, final_positions=None, all_targets=None, episode_targets=None,
                       save_path=None, title="Joystick heatmap (KDE)", show_plot=True,
                       show_target_radius=True, target_radius=None, splash_max=SPLASH_EPISODES_MAX):
    """
    Create a KDE-based heatmap (smoother visualization) with target positions.
    
    Args:
        all_positions: Array of shape (N, 2) with all joystick positions
        final_positions: Array of shape (M, 2) with final positions per episode
        all_targets: Array of shape (K, 2) with all target positions (optional)
        episode_targets: Array of shape (M, 2) with target position per episode (for splash points)
        save_path: Path to save the figure
        title: Title for the plot
        show_plot: Whether to display the plot
    """
    if all_positions is None or len(all_positions) == 0:
        print("  No position data available for heatmap")
        return
    
    # Extract x and y coordinates
    x = all_positions[:, 0]
    y = all_positions[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid for KDE
    x_min, x_max = -1.1, 1.1
    y_min, y_max = -1.1, 1.1
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Compute KDE
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        density = np.reshape(kde(positions).T, xx.shape)
        
        # Plot KDE heatmap
        im = ax.contourf(xx, yy, density, levels=20, cmap='hot', alpha=0.7)
        ax.contour(xx, yy, density, levels=20, colors='black', alpha=0.2, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density', rotation=270, labelpad=20)
    except Exception as e:
        print(f"  KDE computation failed: {e}, using histogram instead")
        # Fallback to histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[-1, 1], [-1, 1]])
        heatmap_log = np.log1p(heatmap)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap_log.T, origin='lower', extent=extent, 
                      cmap='hot', aspect='auto', alpha=0.7, interpolation='bilinear')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Log Density', rotation=270, labelpad=20)
    
    # Add splash points for episode final positions (last N if SPLASH_EPISODES_MAX is set)
    legend_handles = []
    splash_final = _splash_slice(final_positions, splash_max)
    if splash_final is not None:
        n_tot = len(final_positions)
        final_x = splash_final[:, 0]
        final_y = splash_final[:, 1]
        lbl = (
            f'Joystick Final Positions (last {len(splash_final)} of {n_tot})'
            if n_tot > len(splash_final)
            else f'Joystick Final Positions (n={n_tot})'
        )
        scatter1 = ax.scatter(final_x, final_y, c='cyan', s=150, edgecolors='black', 
                  linewidths=2, marker='o', alpha=0.9, 
                  label=lbl,
                  zorder=10)
        legend_handles.append(scatter1)
    
    # Add target positions (if available)
    splash_targets = _splash_slice(episode_targets, splash_max)
    if splash_targets is not None:
        n_tt = len(episode_targets)
        target_x = splash_targets[:, 0]
        target_y = splash_targets[:, 1]
        lbl_t = (
            f'Target Positions (last {len(splash_targets)} of {n_tt})'
            if n_tt > len(splash_targets)
            else f'Target Positions (n={n_tt})'
        )
        scatter2 = ax.scatter(target_x, target_y, c='lime', s=200, edgecolors='darkgreen', 
                  linewidths=2, marker='*', alpha=0.9, 
                  label=lbl_t,
                  zorder=11)
        legend_handles.append(scatter2)
    
    # Add fixed targets with radius circles if enabled
    if show_target_radius:
        from models.callbacks import CurriculumCallback
        fixed_targets = CurriculumCallback.FIXED_TARGETS
        radii = TARGET_RADII_DEFAULT
        mc = FIXED_TARGET_MARKER_COLOR
        rc = FIXED_TARGET_RADIUS_LINE_COLOR

        for i, target in enumerate(fixed_targets):
            ax.scatter(
                target[0],
                target[1],
                c=mc,
                s=200,
                marker="*",
                edgecolors="black",
                linewidths=2,
                zorder=12,
                alpha=0.95,
            )
            for r in radii:
                circle = Circle(
                    target,
                    r,
                    fill=False,
                    color=rc,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.75,
                    zorder=6,
                )
                ax.add_patch(circle)
            ax.text(target[0], target[1] + 0.12, f'T{i+1}', 
                   fontsize=10, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1.0), zorder=25)
    
    if SHOW_SPLASH_LEGEND and legend_handles:
        legend = ax.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=1.0)
        legend.set_zorder(30)
    
    # Add grid and labels
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Joystick X Position (normalized)', fontsize=12)
    ax.set_ylabel('Joystick Y Position (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add center lines
    ax.axhline(0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    
    if SHOW_STATS_TEXT_BOX:
        stats_text = f"Joystick Positions: {len(all_positions):,}\n"
        if final_positions is not None and len(final_positions) > 0:
            stats_text += f"Episodes: {len(final_positions):,}\n"
        if all_targets is not None and len(all_targets) > 0:
            stats_text += f"Target Samples: {len(all_targets):,}"
        if show_target_radius:
            stats_text += "\nTarget Radii: 0.30, 0.20, 0.10, 0.05"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=1.0),
            zorder=30,
        )
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        ensure_dir_for(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved KDE heatmap to {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize joystick movement heatmap with episode final positions"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Path to log directory (required for heatmap modes; optional for --target-single / --targets-blank / --targets-quadrants)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the heatmap image (default: log_dir/joystick_heatmap.png)"
    )
    parser.add_argument(
        "--kde",
        action="store_true",
        help="Use KDE-based heatmap (smoother) instead of histogram"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for histogram heatmap (default: 50)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save)"
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Create a separate heatmap for target positions only"
    )
    parser.add_argument(
        "--targets-blank",
        action="store_true",
        help="Nur leeres Plot der Target-Positionen (kein Heatmap, welche Targets wir geben)"
    )
    parser.add_argument(
        "--targets-space",
        type=str,
        choices=["11", "01"],
        default="11",
        help="Target-Raum für --targets-blank: 11 = [-1,1], 01 = [0,1] (default: 11)"
    )
    parser.add_argument(
        "--radii",
        type=str,
        default="0.1 0.2 0.3",
        help="Radien für --targets-blank / --targets-quadrants (im [-1,1]-Raum), z.B. '0.1 0.2 0.3'"
    )
    parser.add_argument(
        "--targets-quadrants",
        action="store_true",
        help="Leeres Plot der 4 Quadranten-Targets (rechts oben, links unten, rechts unten, links oben)"
    )
    parser.add_argument(
        "--target-single",
        action="store_true",
        help="Nur ein großes Target bei (0,0) mit Pfeil nach innen (Radius decreases, Curriculum)"
    )
    parser.add_argument(
        "--success-csv",
        type=str,
        default=None,
        help="CSV mit target_idx und success (z.B. trials.csv von Fitts-Eval); färbt 20 Targets rot–grün nach Success-Rate"
    )
    parser.add_argument(
        "--show-radius",
        action="store_true",
        default=True,
        help="Show target radius circles around fixed targets (default: True)"
    )
    parser.add_argument(
        "--no-radius",
        action="store_true",
        help="Don't show target radius circles"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Target radius (default: extracted from environment code = 0.5 = 50%%)"
    )
    parser.add_argument(
        "--splash-max",
        type=int,
        default=None,
        help="Max number of episode splash points to plot (default: all)"
    )
    parser.add_argument(
        "--last-positions",
        type=int,
        default=None,
        help="Use only the last N joystick/target samples for heatmap density (default: all)"
    )
    
    args = parser.parse_args()
    
    # Determine if radius should be shown
    show_target_radius = args.show_radius and not args.no_radius

    # Nur ein großes Target + Pfeil nach innen (Curriculum-Veranschaulichung)
    if args.target_single:
        out = args.output or (os.path.join(args.log_dir, "target_single_curriculum.png") if args.log_dir else "target_single_curriculum.png")
        out = add_timestamp_to_filename(out)
        create_target_single_curriculum(save_path=out, title="Target (Curriculum)", show_plot=not args.no_show)
        print("Done!")
        return

    # Nur Blank-Targets-Plot (keine Log-Daten, kein myosuite/env nötig)
    if args.targets_quadrants:
        # 4 Quadranten wie in CurriculumCallback.TARGETS_4_QUADRANTEN ([-1,1]-Raum)
        TARGETS_4 = [
            np.array([0.5, 0.5]),    # rechts oben
            np.array([-0.5, -0.5]),  # links unten
            np.array([0.5, -0.5]),   # rechts unten
            np.array([-0.5, 0.5]),   # links oben
        ]
        targets = [np.asarray(t) for t in TARGETS_4]
        in_01 = args.targets_space == "01"
        if in_01:
            targets = [(t + 1.0) / 2.0 for t in targets]
        radii_list = [float(r) for r in args.radii.split()]
        out = args.output or (os.path.join(args.log_dir, "targets_quadrants_blank.png") if args.log_dir else "targets_quadrants_blank.png")
        out = add_timestamp_to_filename(out)
        create_targets_blank(
            targets,
            save_path=out,
            title="Targets (4 Quadranten)",
            in_01=in_01,
            show_plot=not args.no_show,
            radii=radii_list,
            quadrant_labels=["Q1", "Q2", "Q3", "Q4"],
        )
        print("Done!")
        return

    if args.targets_blank:
        # Gleiche 20 Targets wie in evaluate_fitts.py ([-1,1]-Raum), ohne Env-Import
        TARGETS_20 = [
            np.array([0.82, 0.71]), np.array([-0.91, 0.58]), np.array([0.14, -0.87]), np.array([-0.76, -0.43]),
            np.array([0.61, 0.94]), np.array([-0.38, -0.72]), np.array([0.93, -0.29]), np.array([-0.55, 0.81]),
            np.array([0.27, 0.19]), np.array([-0.84, -0.16]), np.array([0.68, -0.63]), np.array([-0.22, 0.47]),
            np.array([0.45, -0.51]), np.array([-0.67, 0.33]), np.array([0.08, 0.78]), np.array([-0.96, -0.89]),
            np.array([0.53, 0.41]), np.array([-0.49, -0.56]), np.array([0.79, -0.12]), np.array([-0.11, 0.65]),
        ]
        targets = [np.asarray(t) for t in TARGETS_20]
        in_01 = args.targets_space == "01"
        if in_01:
            targets = [(t + 1.0) / 2.0 for t in targets]
        radii_list = [float(r) for r in args.radii.split()]
        success_rates = None
        if args.success_csv and os.path.isfile(args.success_csv):
            import csv as csv_module
            by_target = [[] for _ in range(20)]
            with open(args.success_csv, "r", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    tidx = int(row.get("target_idx", -1))
                    if tidx < 0:
                        continue
                    t20 = tidx % 20
                    s = row.get("success", "0")
                    by_target[t20].append(1 if str(s).strip().lower() in ("true", "1", "yes") else 0)
            success_rates = [float(np.mean(v)) if v else 0.0 for v in by_target]
            print(f"Success rates from {args.success_csv} (per target 0..19): min={min(success_rates):.2f}, max={max(success_rates):.2f}")
        elif args.success_csv:
            print(f"Warning: --success-csv {args.success_csv} not found, showing dummy illustration.")
        if success_rates is None:
            # Dummy success rates for illustration (target sampling: red=low, green=high; small/large circles)
            success_rates = [
                0.25, 0.88, 0.42, 0.72, 0.18, 0.91, 0.55, 0.95, 0.33, 0.62,
                0.22, 0.78, 0.48, 0.85, 0.28, 0.68, 0.52, 0.82, 0.38, 0.75,
            ]
            print("Target sampling illustration: dummy success rates (red=low, green=high; small/large)")
        out = args.output or (os.path.join(args.log_dir, "targets_blank.png") if args.log_dir else "targets_blank.png")
        out = add_timestamp_to_filename(out)
        create_targets_blank(
            targets,
            save_path=out,
            title="Targets (Fitts-Eval)",
            in_01=in_01,
            show_plot=not args.no_show,
            radii=radii_list,
            success_rates=success_rates,
        )
        print("Done!")
        return

    # Heatmap modes require log_dir
    if not args.log_dir:
        parser.error("--log_dir is required for heatmap modes (omit --target-single / --targets-blank / --targets-quadrants to use heatmap)")

    # Get target radius from environment code if not specified (nur für Heatmap-Modi)
    if args.radius is None and show_target_radius:
        from envs.thumb_reach import ThumbReachEnvV0
        args.radius = ThumbReachEnvV0.TARGET_RADIUS
        print(f"Using target radius from code: {args.radius} ({args.radius*100:.0f}%)")

    # Load data
    print("Loading joystick position and target data...")
    all_positions, final_positions, all_targets, episode_targets, metadata = load_joystick_data(args.log_dir)
    final_positions = _skip_first_rows(final_positions, SKIP_FIRST_EPISODES)
    episode_targets = _skip_first_rows(episode_targets, SKIP_FIRST_EPISODES)
    print(f"Ignoring first {SKIP_FIRST_EPISODES} episodes for overlays")
    if args.last_positions is not None:
        all_positions = _last_n_rows(all_positions, args.last_positions)
        all_targets = _last_n_rows(all_targets, args.last_positions)
        final_positions = _last_n_rows(final_positions, args.last_positions)
        episode_targets = _last_n_rows(episode_targets, args.last_positions)
        print(f"Using last {args.last_positions} samples for density and overlays")
    
    if all_positions is None:
        print(" No position data found. Make sure training was run with JoystickTrackerCallback.")
        return
    
    # Determine output path
    if args.output is None:
        output_name = "joystick_heatmap_kde.png" if args.kde else "joystick_heatmap.png"
        args.output = os.path.join(args.log_dir, output_name)
    
    # Add timestamp to output filename
    args.output = add_timestamp_to_filename(args.output)
    
    # Create heatmap
    print("Creating heatmap...")
    fig_title = log_dir_title(args.log_dir, kde=args.kde, targets_only=args.target_only)

    # If target-only mode, create separate target heatmap
    if args.target_only:
        if all_targets is None or len(all_targets) == 0:
            print("  No target data available. Cannot create target-only heatmap.")
            return
        
        # Create target heatmap
        if args.output:
            # Replace the base name with target version and keep timestamp
            directory = os.path.dirname(args.output)
            filename = os.path.basename(args.output)
            # Insert _targets before the timestamp
            if "_" in filename and filename.count("_") >= 3:
                # Has timestamp format: name_YYYY-MM-DD_HH-MM-SS.png
                parts = filename.rsplit("_", 3)
                if len(parts) == 4:
                    base_name, date_part, time_part, ext = parts
                    target_output = os.path.join(directory, f"{base_name}_targets_{date_part}_{time_part}{ext}")
                else:
                    target_output = add_timestamp_to_filename(args.output.replace(".png", "_targets.png"))
            else:
                target_output = add_timestamp_to_filename(args.output.replace(".png", "_targets.png"))
        else:
            target_output = add_timestamp_to_filename(os.path.join(args.log_dir, "target_heatmap.png"))
        
        if args.kde:
            create_kde_heatmap(
                all_targets,
                final_positions=None,
                all_targets=None,
                episode_targets=episode_targets,
                save_path=target_output,
                title=fig_title,
                show_plot=not args.no_show,
                show_target_radius=show_target_radius,
                target_radius=args.radius,
                splash_max=args.splash_max
            )
        else:
            create_heatmap(
                all_targets,
                final_positions=None,
                all_targets=None,
                episode_targets=episode_targets,
                bins=args.bins,
                save_path=target_output,
                title=fig_title,
                show_plot=not args.no_show,
                show_target_radius=show_target_radius,
                target_radius=args.radius,
                splash_max=args.splash_max
            )
    else:
        # Create combined heatmap
        if args.kde:
            create_kde_heatmap(
                all_positions, 
                final_positions,
                all_targets=all_targets,
                episode_targets=episode_targets,
                save_path=args.output,
                title=fig_title,
                show_plot=not args.no_show,
                show_target_radius=show_target_radius,
                target_radius=args.radius,
                splash_max=args.splash_max
            )
        else:
            create_heatmap(
                all_positions, 
                final_positions,
                all_targets=all_targets,
                episode_targets=episode_targets,
                bins=args.bins,
                save_path=args.output,
                title=fig_title,
                show_plot=not args.no_show,
                show_target_radius=show_target_radius,
                target_radius=args.radius,
                splash_max=args.splash_max
            )
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
