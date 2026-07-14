import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import os

# ── DRS Overtaking Model Parameters ──
# Based on real F1 data analysis

# Speed trap delta needed for overtake (km/h)
# Positive = attacking car faster
OVERTAKE_THRESHOLD = 12.0  # km/h minimum delta needed

# Circuit DRS zone lengths (meters)
drs_zones = {
    'Austria': {'length': 630, 'zones': 2},
    'Monza': {'length': 890, 'zones': 2},
    'Monaco': {'length': 180, 'zones': 1},
    'Canada': {'length': 720, 'zones': 2},
    'Spain': {'length': 680, 'zones': 2},
    'Australia': {'length': 570, 'zones': 3},
    'Japan': {'length': 650, 'zones': 2},
}

circuit_colors = {
    'Austria': '#FF3333',
    'Monza': '#DC0000',
    'Monaco': '#FF8700',
    'Canada': '#00AA44',
    'Spain': '#FFF200',
    'Australia': '#00D2BE',
    'Japan': '#0067FF',
}

# ── Overtake Probability Model ──
def overtake_probability(speed_delta, drs_length, tyre_age_diff,
                          defending_tyre='Medium'):
    """
    Calculate probability of successful overtake.

    Parameters:
    - speed_delta: km/h advantage of attacking car at speed trap
    - drs_length: length of DRS zone in meters
    - tyre_age_diff: positive = attacker on fresher tyres
    - defending_tyre: compound of defending car
    """
    # Base probability from speed delta
    # Uses sigmoid function centered at threshold
    base_prob = 1 / (1 + np.exp(-(speed_delta - OVERTAKE_THRESHOLD) / 3))

    # DRS zone length factor (longer = higher probability)
    length_factor = min(drs_length / 700, 1.3)

    # Tyre advantage factor
    tyre_factor = 1 + (tyre_age_diff * 0.008)

    # Defending tyre compound factor
    compound_factors = {
        'Soft': 1.15,    # Fresh softs harder to pass
        'Medium': 1.0,
        'Hard': 0.92,    # Old hards easier to pass
        'UNKNOWN': 1.0
    }
    compound_factor = compound_factors.get(defending_tyre, 1.0)

    # Combined probability
    prob = base_prob * length_factor * tyre_factor / compound_factor
    return np.clip(prob, 0, 0.99)

# ── Scenario Analysis ──
speed_deltas = np.linspace(-5, 30, 200)

# ── Plotting ──
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('F1 DRS Overtaking Probability Model\n2026 Season Analysis',
             fontsize=14, fontweight='bold')

# --- Plot 1: Probability curves by circuit ---
ax1 = axes[0, 0]
for circuit, data in drs_zones.items():
    probs = [overtake_probability(delta, data['length'], 0)
             for delta in speed_deltas]
    ax1.plot(speed_deltas, probs,
             color=circuit_colors[circuit],
             linewidth=2, label=circuit)

ax1.axhline(y=0.5, color='white', linestyle='--',
            linewidth=1, alpha=0.5, label='50% threshold')
ax1.axvline(x=OVERTAKE_THRESHOLD, color='yellow',
            linestyle='--', linewidth=1,
            alpha=0.5, label=f'Min delta ({OVERTAKE_THRESHOLD} km/h)')
ax1.set_xlabel('Speed Trap Delta (km/h)', fontsize=10)
ax1.set_ylabel('Overtake Probability', fontsize=10)
ax1.set_title('Overtake Probability by Circuit\n(Fresh tyres, equal tyre age)',
              fontsize=11)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# --- Plot 2: Tyre age effect ---
ax2 = axes[0, 1]
tyre_age_diffs = [-10, 0, 10, 20, 30]
for age_diff in tyre_age_diffs:
    probs = [overtake_probability(delta, 630, age_diff)
             for delta in speed_deltas]
    label = (f'Attacker {abs(age_diff)} laps fresher'
             if age_diff > 0
             else f'Equal age' if age_diff == 0
             else f'Defender {abs(age_diff)} laps fresher')
    ax2.plot(speed_deltas, probs, linewidth=2, label=label)

ax2.axhline(y=0.5, color='white', linestyle='--',
            linewidth=1, alpha=0.5)
ax2.set_xlabel('Speed Trap Delta (km/h)', fontsize=10)
ax2.set_ylabel('Overtake Probability', fontsize=10)
ax2.set_title('Effect of Tyre Age Difference\n(Austria DRS zone)',
              fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# --- Plot 3: Circuit comparison bar chart ---
ax3 = axes[0, 2]
# At 15 km/h delta (typical DRS advantage)
typical_delta = 15
circuit_probs = {
    circuit: overtake_probability(typical_delta, data['length'], 0)
    for circuit, data in drs_zones.items()
}
circuits_sorted = sorted(circuit_probs.items(),
                          key=lambda x: x[1], reverse=True)

bars = ax3.barh([c[0] for c in circuits_sorted],
                [c[1] for c in circuits_sorted],
                color=[circuit_colors[c[0]] for c in circuits_sorted],
                edgecolor='none', height=0.6)
ax3.set_xlabel('Overtake Probability', fontsize=10)
ax3.set_title(f'Circuit Overtaking Difficulty\n(at {typical_delta} km/h delta)',
              fontsize=11)
ax3.axvline(x=0.5, color='white', linestyle='--',
            linewidth=1, alpha=0.5)
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim(0, 1)

for bar, (circuit, prob) in zip(bars, circuits_sorted):
    ax3.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
             f'{prob:.2f}', va='center', fontsize=9, color='white')

# --- Plot 4: Austrian GP simulation ---
ax4 = axes[1, 0]
np.random.seed(42)

# Simulate 50 DRS attempts at Austrian GP
n_attempts = 50
speed_trap_deltas = np.random.normal(14, 5, n_attempts)
tyre_ages = np.random.randint(-15, 30, n_attempts)
outcomes = []
probs = []

for delta, age in zip(speed_trap_deltas, tyre_ages):
    prob = overtake_probability(delta, 630, age)
    probs.append(prob)
    # Simulate outcome
    outcome = np.random.random() < prob
    outcomes.append(outcome)

successful = sum(outcomes)
colors_scatter = ['#00AA44' if o else '#FF3333' for o in outcomes]

ax4.scatter(speed_trap_deltas, tyre_ages,
            c=colors_scatter, s=80, alpha=0.8, zorder=5)
ax4.axvline(x=OVERTAKE_THRESHOLD, color='yellow',
            linestyle='--', linewidth=1.5,
            label=f'Min threshold ({OVERTAKE_THRESHOLD} km/h)')
ax4.set_xlabel('Speed Trap Delta (km/h)', fontsize=10)
ax4.set_ylabel('Tyre Age Advantage (laps)', fontsize=10)
ax4.set_title(f'Austrian GP DRS Attempt Simulation\n'
              f'{successful}/{n_attempts} successful overtakes '
              f'({successful/n_attempts*100:.0f}%)',
              fontsize=11)

from matplotlib.patches import Patch
legend_elements = [Patch(color='#00AA44', label='Successful'),
                   Patch(color='#FF3333', label='Failed')]
ax4.legend(handles=legend_elements, fontsize=9)
ax4.grid(True, alpha=0.3)

# --- Plot 5: 2D probability heatmap ---
ax5 = axes[1, 1]
delta_range = np.linspace(0, 30, 50)
tyre_range = np.linspace(-20, 30, 50)
delta_grid, tyre_grid = np.meshgrid(delta_range, tyre_range)

prob_grid = np.vectorize(
    lambda d, t: overtake_probability(d, 630, t)
)(delta_grid, tyre_grid)

im = ax5.contourf(delta_grid, tyre_grid, prob_grid,
                   levels=20, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(im, ax=ax5, label='Overtake Probability')
ax5.contour(delta_grid, tyre_grid, prob_grid,
            levels=[0.5], colors='white',
            linewidths=2, linestyles='--')
ax5.set_xlabel('Speed Trap Delta (km/h)', fontsize=10)
ax5.set_ylabel('Tyre Age Advantage (laps)', fontsize=10)
ax5.set_title('Overtake Probability Heatmap\nSpeed Delta vs Tyre Advantage',
              fontsize=11)
ax5.text(16, 20, '50%\nthreshold', color='white',
         fontsize=9, ha='center')

# --- Plot 6: Season overtake probability by race ---
ax6 = axes[1, 2]
races_2026 = list(drs_zones.keys())
race_probs = []

for race in races_2026:
    # Simulate average conditions
    avg_prob = overtake_probability(14, drs_zones[race]['length'], 5)
    race_probs.append(avg_prob * drs_zones[race]['zones'])

colors_race = [circuit_colors[r] for r in races_2026]
bars6 = ax6.bar(races_2026, race_probs,
                color=colors_race, edgecolor='none', alpha=0.85)
ax6.set_ylabel('Expected Overtakes per DRS Zone\n(per attempt)', fontsize=9)
ax6.set_title('Overtaking Opportunity Index\nby Circuit (2026 season)',
              fontsize=11)
ax6.set_xticklabels(races_2026, rotation=30, ha='right', fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars6, race_probs):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('drs_overtaking.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== DRS OVERTAKING PROBABILITY REPORT ===")
print(f"Overtake threshold: {OVERTAKE_THRESHOLD} km/h")
print(f"\nCircuit Rankings (at 15 km/h delta):")
for circuit, prob in circuits_sorted:
    difficulty = 'Easy' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Hard'
    print(f"  {circuit}: {prob:.2f} ({difficulty})")
print(f"\nAustrian GP Simulation:")
print(f"  Total DRS attempts: {n_attempts}")
print(f"  Successful overtakes: {successful}")
print(f"  Success rate: {successful/n_attempts*100:.1f}%")
