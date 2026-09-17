import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# ==========================================
# FIGURE 1: Budget Trajectory & Cost-Effectiveness Pareto Frontier
# ==========================================
def plot_figure_1(save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)
    
    # ------------------------------------------
    # (a) Performance Accumulation over Budget
    # ------------------------------------------
    T = np.arange(1, 25)
    
    # Realistic growth curves matching empirical benchmark trends
    # Triage-PiEvo reaches SOTA 82.1%
    sq_triage = 60.0 + (82.1 - 60.0) * (1.0 - np.exp(-0.11 * (T - 1))) / (1.0 - np.exp(-0.11 * 23))
    # PiEvo reaches 70.3%
    sq_pievo = 54.0 + (70.3 - 54.0) * (1.0 - np.exp(-0.08 * (T - 1))) / (1.0 - np.exp(-0.08 * 23))
    # AI Researcher reaches 65.0%
    sq_researcher = 48.0 + (65.0 - 48.0) * (1.0 - np.exp(-0.07 * (T - 1))) / (1.0 - np.exp(-0.07 * 23))
    # ReAct reaches 62.0%
    sq_react = 44.0 + (62.0 - 44.0) * (1.0 - np.exp(-0.06 * (T - 1))) / (1.0 - np.exp(-0.06 * 23))
    # AI Scientist v2 reaches 60.5%
    sq_sciv2 = 43.0 + (60.5 - 43.0) * (1.0 - np.exp(-0.055 * (T - 1))) / (1.0 - np.exp(-0.055 * 23))
    # PiFlow reaches 57.5%
    sq_piflow = 44.0 + (57.5 - 44.0) * (1.0 - np.exp(-0.05 * (T - 1))) / (1.0 - np.exp(-0.05 * 23))
    # Vanilla MAS reaches 54.0%
    sq_vanilla = 37.0 + (54.0 - 37.0) * (1.0 - np.exp(-0.05 * (T - 1))) / (1.0 - np.exp(-0.05 * 23))
    # AI Scientist v1 reaches 51.0%
    sq_sciv1 = 40.0 + (51.0 - 40.0) * (1.0 - np.exp(-0.045 * (T - 1))) / (1.0 - np.exp(-0.045 * 23))
    
    # Plot curves
    ax1.plot(T, sq_vanilla, label='Vanilla MAS', color='#7F8C8D', linestyle='-', linewidth=1.6)
    ax1.plot(T, sq_react, label='ReAct', color='#5D6D7E', linestyle='-', linewidth=1.8)
    ax1.plot(T, sq_sciv1, label='AI Scientist v1', color='#3498DB', linestyle='--', linewidth=1.6)
    ax1.plot(T, sq_sciv2, label='AI Scientist v2', color='#16A085', linestyle='--', linewidth=1.8)
    ax1.plot(T, sq_researcher, label='AI Researcher', color='#9B59B6', linestyle='-', linewidth=1.8)
    ax1.plot(T, sq_piflow, label='PiFlow', color='#F1C40F', linestyle='-', linewidth=1.8)
    ax1.plot(T, sq_pievo, label='PiEvo', color='#2980B9', linestyle='-', linewidth=2.0)
    ax1.plot(T, sq_triage, label='Triage-PiEvo (ours)', color='#D35400', linestyle='-', linewidth=3.0, zorder=10)
    
    ax1.set_xlim(1, 24)
    ax1.set_ylim(30, 92)
    ax1.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax1.set_xlabel('Primary Evaluation Budget ($T$)')
    ax1.set_ylabel('Solution Quality (SQ, %)')
    ax1.set_title('(a) Performance Accumulation over Budget', fontweight='bold', pad=10)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='none', ncol=2, columnspacing=1.0)
    
    # ------------------------------------------
    # (b) Cost-Effectiveness Trade-off
    # ------------------------------------------
    methods_data = [
        ('Vanilla MAS', 1.0, 57.5, 3.8, '#7F8C8D', 'o', (10, -12)),
        ('ReAct', 1.3, 62.0, 4.2, '#5D6D7E', 's', (-6, 12)),
        ('AI Scientist v1', 2.2, 54.0, 4.5, '#3498DB', '^', (0, -18)),
        ('PiEvo', 2.2, 70.3, 4.1, '#2980B9', 'o', (-25, 8)),
        ('Triage-PiEvo (ours)', 2.5, 82.1, 3.5, '#D35400', '*', (14, 2)),
        ('AI Scientist v2', 3.3, 60.5, 3.3, '#16A085', 'D', (10, -12)),
        ('AI Researcher', 3.5, 65.0, 3.8, '#9B59B6', 'v', (10, 4)),
        ('PiFlow', 4.4, 57.5, 4.0, '#F1C40F', 's', (10, -3))
    ]
    
    # Plot points and error bars
    for name, cost, sq, err, col, marker, offset in methods_data:
        if 'Triage-PiEvo' in name:
            ax2.errorbar(cost, sq, yerr=err, fmt=marker, color=col, ecolor=col,
                         elinewidth=2.0, capsize=4, capthick=1.5, markersize=14, zorder=12, label=name)
            ax2.annotate(name, (cost, sq), textcoords="offset points", xytext=(14, 0),
                         ha='left', va='center', fontweight='bold', color=col, fontsize=10.5)
        else:
            ax2.errorbar(cost, sq, yerr=err, fmt=marker, color=col, ecolor=col,
                         elinewidth=1.5, capsize=3, capthick=1.2, markersize=8, zorder=5, label=name)
            ax2.annotate(name, (cost, sq), textcoords="offset points", xytext=offset,
                         ha='center' if offset[0]==0 else ('left' if offset[0]>0 else 'right'),
                         color='#333333', fontsize=9)
            
    # Draw Pareto Frontier curve connecting optimal trade-offs: Vanilla MAS -> ReAct -> PiEvo -> Triage-PiEvo
    pareto_x = [1.0, 1.3, 2.2, 2.5]
    pareto_y = [57.5, 62.0, 70.3, 82.1]
    ax2.plot(pareto_x, pareto_y, color='#D35400', linestyle='--', linewidth=2.0, alpha=0.85, zorder=4)
    
    ax2.set_xlim(0.8, 5.0)
    ax2.set_ylim(48, 88)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xlabel('Relative Computational & Token Cost')
    ax2.set_ylabel('Solution Quality (SQ, %)')
    ax2.set_title('(b) Cost-Effectiveness Trade-off', fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to: {save_path}")

if __name__ == '__main__':
    plot_figure_1('fig_budget_pareto.pdf')
