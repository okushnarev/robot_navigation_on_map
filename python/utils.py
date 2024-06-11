import matplotlib
from matplotlib import pyplot as plt


def init_matplotlib():
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update({
        'savefig.dpi':     800,
        'font.family':     'Serif',
        'font.serif':      'Times New Roman',
        'font.size':       16,
        'axes.axisbelow':  True,

        "axes.labelpad":   12,
        "axes.labelsize":  14,

        'legend.fontsize': 14,

        "xtick.bottom":    False,
        "ytick.left":      False,

        "xtick.major.pad": 10,
        "ytick.major.pad": 15,
    })


def to_grid(coeff):
    def fun(val):
        return int(val * coeff)

    return fun
