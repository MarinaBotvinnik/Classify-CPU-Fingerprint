import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import glob

# which archs exist?
print('finding all data to plot..')
datafiles = glob.glob('train/Gen10-4/mean_map.npy')
if len(datafiles) < 1:
    raise Exception('no data files found to plot')
    sys.exit(1)
print('found:', datafiles)

archs = []
arch2name = {'SKL': 'Intel Skylake', 'BDW': 'Intel Broadwell Xeon', 'ZEN+': 'AMD EPYC ZEN+', 'KBL': 'Intel Kabylake', 'ICL': 'Intel Icelake'}

for datafile in datafiles:
    parts = datafile.split('_')
    this_arch = 'ICL'
    print('datafile: %s arch: %s' % (datafile, this_arch))
    archs.append(this_arch)
    if this_arch not in arch2name:
        print('WARNING: no human-friendly name for %s' % this_arch)

nsubplots = len(archs)

fig, axes = plt.subplots(nrows=1, ncols=nsubplots, figsize=(12, 9))
plt.axis('off')

assert nsubplots > 0
if nsubplots == 1:
    axes = (axes,)

for (arch, ax) in zip(archs, axes):
    data = numpy.load('train/Gen10-4/mean_map.npy')
    im = ax.pcolormesh(data, vmin=1.0, vmax=3.0)

fig.patch.set_visible(False)
print('saving png')
plt.savefig('covert-nuke2.png', bbox_inches='tight')
print('saving done')