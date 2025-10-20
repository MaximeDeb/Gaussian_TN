import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cmap = matplotlib.colormaps['afmhot']
#cmap = matplotlib.colormaps['plasma']
#cmap = matplotlib.colormaps['inferno']
colors__ = [cmap(n) for n in np.linspace(0, 1, 9)]   


##-----------------
def prepare_standard_figure(nrows=1, ncols=1, sharex=False, sharey=False, width=3.375, aspect_ratio=1.61,colourmap=cmap,nlines=9):

    plt.rc('text', usetex=True)
    #plt.rc('text.latex', preamble=r'\usepackage{amsmath},'+r'\usepackage{sfmath}')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='serif', size=7)
    plt.rc('lines', linewidth=1.0)
    plt.rc('xtick', labelsize='medium', direction="out", top=False)
    plt.rc('ytick', labelsize='medium', direction="out", right=False)
    plt.rc('legend', fontsize='small', numpoints=1, frameon=False, handlelength=1)
    plt.rc('axes', linewidth=0.5)
    plt.rc('errorbar', capsize=1)
    plt.rc('savefig', dpi=300)

    cmap = matplotlib.cm.get_cmap(colourmap)
    colors__ = [cmap(n) for n in np.linspace(0, 1, nlines)]
    # change color scheme
    #colors__ = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"]
    plt.rcParams['axes.prop_cycle'] = cycler.cycler(color=colors__)

    fig_size = (width, width/aspect_ratio)
    f1, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=fig_size)
    return f1, axs

##-----------------
def add_text(ax, x, y, txt, color=0.5, fontsize="small", alpha=1, rotation=0):
	if type(color)!=float:
		color=color
	else:
		color=cmap(color)
	ax.text(x, y, txt, color=color, fontsize=fontsize, alpha=alpha, rotation=rotation)
	return

##-----------------
def add_hlines(ax, y, x0, x1, lw=1., ls='solid', color=0.5, fontsize="small", alpha=1):
	if type(color)!=float:
		color=color
	else:
		color=cmap(color)
	ax.hlines(y, x0, x1, lw=lw, ls=ls, color=color, alpha=alpha)
	return

##-----------------
def add_inset(ax, width="100%", height="100%", bbox_to_anchor=(1e-2, 2, 1e3, 3),  loc="upper left", borderpad=0):
	ax_inset = inset_axes(ax, width=width, height=height, bbox_to_anchor=bbox_to_anchor, loc=loc, borderpad=borderpad)
	return ax_inset

##-----------------
def plot1d(ax, x, y, label=None, lw=0.5, ls='solid', marker='+', markersize = 3, color=0.5, alpha=1, fillstyle='full', mew=1):
	if type(color)!=float:
		color=color
	else:
		color=cmap(color)
	ax.plot(x, y, lw=lw, marker=marker, markersize=markersize, alpha=alpha, color=color, label=label, ls=ls, fillstyle=fillstyle, markeredgewidth=mew)
	#ax.set(yticks=(np.linspace(0,round(max(y),2),3) ), yticklabels=(np.linspace(0,round(max(y),2),3) ), xticks=(np.linspace(0,round(max(x),2)*100,3)/100).astype(int) )
	#ax.set(yticks=(np.linspace(0,round(max(y),2),3) ), yticklabels=(np.linspace(0,round(max(y),2),3) ), xticks=(np.linspace(0,(max(x),3)).astype(int) ) )
	ax.legend(fontsize='small', numpoints=1, frameon=False, handlelength=1)
	return 

##-----------------
def heat_map(ax, matrix, yaxis, cmap=cmap, npt_y=10, alpha=None, vmin=None, vmax=None, aspect="equal", origin="lower", bar=True): ## OR USE PCOLORMESH
	heatmap = ax.imshow(matrix, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax, alpha=alpha, origin=origin, interpolation='none' )
	if bar:
		plt.colorbar(heatmap, ax=ax, cmap=cmap)
	#ax.set(yticks=np.linspace(0,int(matrix.shape[0]),3), yticklabels=(np.linspace(0,int(matrix.shape[0]),3) * 0.05).astype(int), xticks=np.linspace(0,int(matrix.shape[1]),3).astype(int))
	#ax.set(yticks=np.linspace(0,matrix.shape[0], npt_y), yticklabels=yaxis[::len(yaxis)//npt_y].astype(int))
	return heatmap
