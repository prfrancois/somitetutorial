import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as integrate
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#Making a nice movie of beads in a landscape                   
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


pi = np.pi

size=15
params = {'legend.fontsize': 'large',
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,
          'ytick.labelsize': size,
          'axes.titlepad': 25,
          'lines.linewidth': 3,
          'figure.dpi': 100}
plt.rcParams.update(params)


#General Utilities to present kymos

def plot_kymo(result, title=None, ax=None, colorbar=False, vmin=None, vmax=None, cb_ticks=None, interpolation=None):
    show = False
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = (5,5))
        show = True
    kymo = ax.imshow(result, origin = 'lower', aspect='auto', vmin=vmin, vmax=vmax, interpolation=interpolation)
    if colorbar:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="5%")
        extend='neither'
        if vmin is not None and vmax is not None and cb_ticks is not None:
            if max(cb_ticks) > vmax and min(cb_ticks) < vmin:
                extend='both'
            elif max(cb_ticks) > vmax:
                extend='max'
            elif min(cb_ticks) < vmin:
                extend='min'
        cb = plt.colorbar(kymo, cax=cax, extend=extend, ticks=cb_ticks)
        cb.outline.set_visible(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.spines['left'].set_visible(False)
    
    ax.axes.spines['right'].set_visible(False)
    ax.axes.spines['top'].set_visible(False)
    ax.axes.spines['bottom'].set_visible(False)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Space')
    ax.set_title(title)

    if show:
        plt.show()


#1 D diffusion operator
def diffusion(vec, h=1., boundary_derivs=(0., 0.)):
    dif = np.zeros(len(vec))
    dif[0] =  vec[1] - 2*vec[0] + (vec[0] - boundary_derivs[0]*h)
    dif[-1] = (vec[-1] + boundary_derivs[1]*h) - 2.*vec[-1] + vec[-2]
    dif[1:-1] = (np.roll(vec, -1) - 2.*vec + np.roll(vec, +1))[1:-1]
    return dif/h**2



#Functions for flow plots and to show excitable dynamics

def phase_plot(L_1,L_2,npoints,some_derivs,*args):
    q = np.linspace(L_1, L_2, npoints)
    xx, yy = np.meshgrid(q,q, indexing = 'xy')
    dZ=some_derivs(1,np.array([xx,yy]), *args)
    vec = np.reshape(dZ, (2,npoints,npoints))   
    fig, ax = plt.subplots(1, figsize = (10,10))
    dX=vec[0,:]
    dY=vec[1,:]
    ax.streamplot(xx, yy, dX, dY, density = 3, color = 'gray', linewidth = 0.5)
    ax.contour(xx, yy, dX, (0,), colors=('royalblue',), linewidths=5, alpha=.7) # plots x-nullcline
    ax.contour(xx, yy, dY, (0,), colors=('green',), linewidths=5, alpha=.7) # y-nullcline
    return fig,ax


def plot_excitable(some_derivs,init_vec,L,*args):

    t = np.linspace(0., 200., 5000)
    ap_positions = np.linspace(0.0, 1.0, 1)
    names_plot = ['A', 'S']
    colors_plot = ['tab:red', 'tab:blue']
    results_osc = integrate(some_derivs, (np.min(t), np.max(t)), init_vec, method='RK45', t_eval=t, args=args, rtol=1e-5).y
    plt.figure(figsize = (8,4))
    for i in range(2):
        plt.plot(t/100, results_osc[i], c = colors_plot[i], label = names_plot[i])
    plt.legend()
    plt.xlabel('Time [a.u.]')
    plt.ylabel('Oscillation [a.u.]')
    plt.show()
    npoints=50
    fig,ax=phase_plot(-L,L,npoints,some_derivs,*args)
    ax.plot(results_osc[0],results_osc[1], color='r')
    return fig,ax