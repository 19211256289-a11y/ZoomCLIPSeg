"""Shared figure style: 8 pt sans, vector PDF with embedded TrueType, 600 dpi PNG."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID = '#1a1a19', '#52514e', '#e4e3df'
BASE, DETAIL, ZOOM = '#7a7974', '#2a78d6', '#eb6834'   # neutral reference, categorical slots 1 and 2
SINGLE, TWO = DETAIL, ZOOM
ONE_COL, TWO_COL = 3.42, 7.08   # inches: column and text width of the two-column layout (87 mm, 180 mm)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'figure.dpi': 150,
    'mathtext.fontset': 'dejavusans',
})


def save(fig, name, exact=False):
    """exact=True keeps the figure at its nominal size (use with constrained layout), so that the 8 pt labels are
    8 pt in print when the file is included at its natural width."""
    kw = dict(bbox_inches=None) if exact else {}
    fig.savefig(f'figures/{name}.pdf', **kw)
    fig.savefig(f'figures/{name}.png', dpi=600, **kw)
    plt.close(fig)


def panel(ax, letter, title=None):
    ax.text(0, 1.04, f'({letter})', transform=ax.transAxes, fontweight='bold', fontsize=8.5, va='bottom', ha='left')
    if title:
        ax.text(0.085, 1.04, title, transform=ax.transAxes, fontsize=8.5, va='bottom', ha='left')
