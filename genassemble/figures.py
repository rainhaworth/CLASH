import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib.font_manager as fm
from tqdm import tqdm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Nimbus Roman'

mega_files = glob.glob('mega*.csv')
nucm_files = glob.glob('nucm*.csv')
params_file = 'params.csv'

# read params
print('reading params')
with open(params_file, 'r') as f:
    # get line -> split by ',' -> remove empty field
    lengths = f.readline().split(',')[:-1]
    psubs = f.readline().split(',')[:-1]
    eirs = f.readline().split(',')[:-1]
    eiss = f.readline().split(',')[:-1]

# grid params
nrow = len(lengths)
ncol = len(psubs)

for tool in ['mega', 'nucm']:
    # grid subplots
    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)

    # find min and max
    # there's probably a better way to write this but i have covid so this is fine
    # we need to pre-compute globals so that the heatmap colors are all the same
    minval = np.inf
    maxval = -np.inf
    for length in lengths:
        for psub in psubs:
            fn = tool + '-l' + str(length) + '-p' + str(psub) + '.csv'
            data = np.genfromtxt(fn, delimiter=',', usecols=range(len(eirs)))

            _max = np.max(data)
            _min = np.min(data)
            if _max > maxval:
                maxval = _max
            if _min < minval:
                minval = _min

    # draw all plots
    print('plotting', tool)
    for i, length in tqdm(enumerate(lengths), total=len(lengths)):
        for j, psub in enumerate(psubs):
            ax_curr = axs[i,j]
            
            # plot
            fn = tool + '-l' + str(length) + '-p' + str(psub) + '.csv'
            data = np.genfromtxt(fn, delimiter=',', usecols=range(len(eirs)))

            sns.heatmap(data, vmin=minval, vmax=maxval,
                        cbar=False, square=True, cmap='crest',
                        ax=ax_curr, xticklabels=eiss, yticklabels=eirs)

            # hide ticks + labels
            ax_curr.tick_params(axis='both', which='both', bottom=False, left=False, labelsize='xx-small')
                                #, labelbottom=False, labelleft=False)

            # labels
            if j == 0:
                ax_curr.set_ylabel(length)
            if i == len(lengths)-1:
                ax_curr.set_xlabel(psub)

    name = ''
    if tool == 'mega':
        name = 'MEGAHIT'
    elif tool == 'nucm':
        name = 'MUMmer'

    fig.suptitle(name + ' Score, Indel Rate vs. Size')
    fig.supxlabel('P(sub)')
    fig.supylabel('Length')
    plt.tight_layout()

    # do this after tight_layout or figure breaks
    fig.colorbar(axs[0, 0].collections[0], ax=axs)

    plt.savefig(tool + '-grid.png', dpi=200)
    plt.cla()