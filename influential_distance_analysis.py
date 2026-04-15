"""
influential_distance_analysis.py
=================================
Estimates a physically grounded value for `influential_distance` in the
PELT+ + Graph shockwave detector by analysing car-following platoon
cumulative spacing in NGSIM data.

`influential_distance` is defined as the **per-vehicle reaction reach**:
the spatial distance over which a single vehicle's deceleration event is
likely to be felt by its upstream neighbours.  Empirically, this equals
the cumulative bumper-to-bumper gap across a short platoon of N followers
plus the vehicle body lengths that occupy that stretch of road.

Usage
-----
    python influential_distance_analysis.py \
        --data_path /path/to/final_US101_trajectories-0805am-0820am.csv \
        --platoon_size 3          # adjustable N-vehicle following chain
        --lane_id 2               # optional: restrict to one lane (0 = all)
        --duration 0 900          # optional: time window [start end] in seconds
        --output_dir ./results
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── reuse existing loader ────────────────────────────────────────────────────
# Add project root to path so dataloader can be found whether the script is
# executed from the project directory or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import NGSIMDataLoader          # noqa: E402  (project module)


# ════════════════════════════════════════════════════════════════════════════
# Core analysis
# ════════════════════════════════════════════════════════════════════════════

def build_platoon_chains(raw_df: pd.DataFrame,
                         platoon_size: int,
                         min_frames: int = 30) -> pd.DataFrame:
    """
    For every vehicle that has a valid `Preceeding` pointer, trace the
    car-following chain forward (leader → follower 1 → … → follower N-1)
    and collect cumulative spatial headway.

    Parameters
    ----------
    raw_df      : raw NGSIM DataFrame (already loaded, motorcycles removed)
    platoon_size: N — number of vehicles in the following chain
                  (1 means just leader→1st follower; 3 means leader + 3
                  downstream vehicles)
    min_frames  : minimum coincident frames for a valid platoon observation

    Returns
    -------
    DataFrame with one row per (leader, frame) pair that has a complete chain.
    Columns:
        leader_id, frame, lane, cum_space_hdwy_ft,
        mean_space_hdwy_ft, n_vehicles
    """
    if platoon_size < 1:
        raise ValueError("platoon_size must be >= 1")

    print(f"\n[build_platoon_chains] platoon_size={platoon_size}, "
          f"analysing {len(raw_df):,} records …")

    # Index once by (Vehicle_ID, Frame_ID) for O(1) look-ups
    df = raw_df.copy()
    df = df[df['Space_Hdwy'] > 0]          # rows with valid headway only
    df.set_index(['Vehicle_ID', 'Frame_ID'], inplace=True)
    df.sort_index(inplace=True)

    # Build a quick {(veh, frame): preceeding_id} lookup
    prec_lookup = df['Preceeding'].to_dict()   # key=(veh,frame)
    hdwy_lookup  = df['Space_Hdwy'].to_dict()  # key=(veh,frame)
    lane_lookup  = df['Lane_ID'].to_dict()

    records = []

    # Iterate over all vehicles that appear in the index as "followers"
    veh_frames = df.index.to_list()          # list of (veh_id, frame_id)

    for (follower_id, frame_id) in veh_frames:
        # Walk UPSTREAM: follower → Preceeding → Preceeding → …
        chain_hdwy = []
        current_veh = follower_id
        valid = True

        for _ in range(platoon_size):
            key = (current_veh, frame_id)
            prec = prec_lookup.get(key, 0)
            hdwy = hdwy_lookup.get(key, 0.0)

            if prec == 0 or hdwy <= 0:
                valid = False
                break

            chain_hdwy.append(hdwy)
            current_veh = prec          # move to next upstream vehicle

        if not valid or len(chain_hdwy) < platoon_size:
            continue

        lane_id = lane_lookup.get((follower_id, frame_id), -1)
        cum_hdwy = float(np.sum(chain_hdwy))
        mean_hdwy = float(np.mean(chain_hdwy))

        records.append({
            'leader_id':          current_veh,
            'follower_id':        follower_id,
            'frame':              frame_id,
            'lane':               lane_id,
            'cum_space_hdwy_ft':  cum_hdwy,
            'mean_space_hdwy_ft': mean_hdwy,
            'n_vehicles':         platoon_size,
        })

    chains = pd.DataFrame(records)
    print(f"  → {len(chains):,} valid platoon observations collected")
    return chains


def summarise_influential_distance(chains: pd.DataFrame,
                                   percentiles: list | None = None
                                   ) -> dict:
    """
    Compute descriptive statistics for `cum_space_hdwy_ft` and recommend
    a value for `influential_distance`.

    Strategy
    --------
    The P85 (85th-percentile) of cumulative headway is used as the
    recommended value.  This mirrors the 85th-percentile design speed logic
    in traffic engineering: it covers the *typical* congested case without
    being dominated by extreme outliers (e.g. a single truck stopped far
    from traffic).

    Returns a dict with keys: mean, median, std, p50, p75, p85, p95,
    recommended_ft, recommended_m.
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 85, 90, 95]

    vals = chains['cum_space_hdwy_ft'].dropna().values
    pct_vals = np.percentile(vals, percentiles)
    pct_dict = {f'p{p}': float(v) for p, v in zip(percentiles, pct_vals)}

    recommended_ft = pct_dict['p85']
    recommended_m  = recommended_ft * 0.3048

    stats = {
        'n_observations':  len(vals),
        'mean':            float(np.mean(vals)),
        'median':          float(np.median(vals)),
        'std':             float(np.std(vals)),
        **pct_dict,
        'recommended_ft':  recommended_ft,
        'recommended_m':   recommended_m,
    }
    return stats


# ════════════════════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════════════════════

def plot_results(chains: pd.DataFrame,
                 stats: dict,
                 platoon_size: int,
                 output_dir: Path) -> None:
    """
    Produce a two-panel figure:
      Left  – histogram + KDE of cumulative headway with percentile markers
      Right – per-lane box-plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Influential Distance Analysis  |  Platoon size N = {platoon_size}",
        fontsize=13, fontweight='bold'
    )

    # ── Left: histogram ─────────────────────────────────────────────────────
    ax = axes[0]
    vals = chains['cum_space_hdwy_ft'].dropna().values

    ax.hist(vals, bins=60, color='steelblue', alpha=0.65,
            edgecolor='white', linewidth=0.4, density=True,
            label='Observed (density)')

    # KDE overlay (manual Gaussian kernel to avoid scipy dependency)
    bw = 1.06 * vals.std() * len(vals) ** (-1/5)   # Silverman bandwidth
    x_range = np.linspace(max(0, vals.min() - bw * 3),
                          vals.max() + bw * 3, 400)
    kde = np.mean(
        np.exp(-0.5 * ((x_range[:, None] - vals[None, :]) / bw) ** 2),
        axis=1
    ) / (bw * np.sqrt(2 * np.pi))
    ax.plot(x_range, kde, color='navy', lw=1.8, label='KDE')

    # Percentile markers
    colours = {'p50': '#2ca02c', 'p75': '#ff7f0e',
               'p85': '#d62728', 'p95': '#9467bd'}
    labels  = {'p50': 'P50', 'p75': 'P75',
               'p85': 'P85 (recommended)', 'p95': 'P95'}
    for key, col in colours.items():
        xv = stats[key]
        ax.axvline(xv, color=col, ls='--', lw=1.6,
                   label=f'{labels[key]} = {xv:.0f} ft')

    ax.set_xlabel('Cumulative space headway across platoon (ft)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Distribution (N={stats["n_observations"]:,} obs.)', fontsize=11)
    ax.legend(fontsize=8.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    ax.grid(axis='y', alpha=0.3)

    # ── Right: per-lane box-plots ────────────────────────────────────────────
    ax2 = axes[1]
    lanes = sorted(chains['lane'].unique())
    data_per_lane = [
        chains.loc[chains['lane'] == l, 'cum_space_hdwy_ft'].dropna().values
        for l in lanes
    ]
    bp = ax2.boxplot(data_per_lane, labels=[f'Lane {l}' for l in lanes],
                     patch_artist=True,
                     medianprops=dict(color='black', lw=1.8),
                     whiskerprops=dict(lw=1.2),
                     capprops=dict(lw=1.2))

    palette = plt.cm.tab10.colors
    for patch, col in zip(bp['boxes'], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    ax2.axhline(stats['p85'], color='#d62728', ls='--', lw=1.5,
                label=f'P85 = {stats["p85"]:.0f} ft')
    ax2.set_xlabel('Lane', fontsize=11)
    ax2.set_ylabel('Cumulative space headway (ft)', fontsize=11)
    ax2.set_title('Per-lane distribution', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f'influential_distance_N{platoon_size}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[plot] Figure saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def print_report(stats: dict, platoon_size: int) -> None:
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  INFLUENTIAL DISTANCE REPORT  (platoon_size = {platoon_size})")
    print(sep)
    print(f"  Observations : {stats['n_observations']:>10,}")
    print(f"  Mean         : {stats['mean']:>10.1f} ft   "
          f"({stats['mean'] * 0.3048:.1f} m)")
    print(f"  Median (P50) : {stats['median']:>10.1f} ft   "
          f"({stats['median'] * 0.3048:.1f} m)")
    print(f"  Std dev      : {stats['std']:>10.1f} ft")
    print(f"  P75          : {stats['p75']:>10.1f} ft   "
          f"({stats['p75'] * 0.3048:.1f} m)")
    print(f"  P85          : {stats['p85']:>10.1f} ft   "
          f"({stats['p85'] * 0.3048:.1f} m)")
    print(f"  P95          : {stats['p95']:>10.1f} ft   "
          f"({stats['p95'] * 0.3048:.1f} m)")
    print(sep)
    print(f"  ★ Recommended influential_distance  (P85)")
    print(f"      = {stats['recommended_ft']:.1f} ft  ≈  "
          f"{stats['recommended_m']:.1f} m")
    print(sep)
    print()


def save_csv(chains: pd.DataFrame, stats: dict,
             platoon_size: int, output_dir: Path) -> None:
    # Per-observation CSV
    obs_path = output_dir / f'platoon_chains_N{platoon_size}.csv'
    chains.to_csv(obs_path, index=False)
    print(f"[save] Platoon chain data → {obs_path}")

    # Summary CSV
    summary = {k: [v] for k, v in stats.items()}
    summary['platoon_size'] = [platoon_size]
    pd.DataFrame(summary).to_csv(
        output_dir / f'influential_distance_summary_N{platoon_size}.csv',
        index=False
    )
    print(f"[save] Summary statistics → "
          f"{output_dir / f'influential_distance_summary_N{platoon_size}.csv'}")


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate influential_distance from NGSIM platoon spacing"
    )
    p.add_argument(
        '--data_path', type=str,
        default='../Dataset/final_US101_trajectories-0805am-0820am.csv',
        help='Path to NGSIM CSV file'
    )
    p.add_argument(
        '--platoon_size', type=int, default=2,
        help='Number of following vehicles in platoon chain (default: 3)'
    )
    p.add_argument(
        '--lane_id', type=int, default=0,
        help='Lane to restrict analysis to (0 = all lanes, default: 0)'
    )
    p.add_argument(
        '--duration', type=float, nargs=2, default=None,
        metavar=('START', 'END'),
        help='Time window in seconds, e.g. --duration 0 900'
    )
    p.add_argument(
        '--output_dir', type=str, default='./results',
        help='Directory for output figures and CSVs (default: ./results)'
    )
    p.add_argument(
        '--no_plot', action='store_true',
        help='Skip figure generation'
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data via existing NGSIMDataLoader ────────────────────────────
    print(f"\n[load] {args.data_path}")
    loader = NGSIMDataLoader(args.data_path)
    duration = tuple(args.duration) if args.duration else None
    raw_df = loader.load_data(duration=duration)

    # Optionally restrict to one lane
    if args.lane_id > 0:
        raw_df = raw_df[raw_df['Lane_ID'] == args.lane_id].copy()
        print(f"[filter] Restricted to Lane {args.lane_id}: "
              f"{len(raw_df):,} records remain")

    # ── 2. Build platoon chains ──────────────────────────────────────────────
    chains = build_platoon_chains(raw_df, platoon_size=args.platoon_size)

    if chains.empty:
        print("\n[ERROR] No valid platoon chains found.  "
              "Check that Space_Hdwy and Preceeding fields are populated "
              "for the selected lane / time window.")
        return

    # ── 3. Summarise ────────────────────────────────────────────────────────
    stats = summarise_influential_distance(chains)
    print_report(stats, args.platoon_size)

    # ── 4. Save outputs ──────────────────────────────────────────────────────
    save_csv(chains, stats, args.platoon_size, output_dir)

    if not args.no_plot:
        plot_results(chains, stats, args.platoon_size, output_dir)

    return stats


if __name__ == '__main__':
    main()
