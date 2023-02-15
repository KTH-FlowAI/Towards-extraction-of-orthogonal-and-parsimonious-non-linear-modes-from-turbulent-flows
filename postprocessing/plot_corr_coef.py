#%%
import numpy as np
from matplotlib import pyplot as plt
from error import err
from scipy.io import loadmat
import cmocean.cm as cmo
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 14)
plt.rc('legend', fontsize = 12, handletextpad=0.3)              
plt.rc('xtick', labelsize = 10)             
plt.rc('ytick', labelsize = 10)

ddir = './pred_data/'

bvae_name = "VAE_ld5_b1e-3"
ae_name = "AE_ld5"
hae_name = "HAE_5"
pod_name = "POD-5"

r = 5 
d_vae = np.load(ddir+bvae_name+'.npz')
d_ae = np.load(ddir+ae_name+'.npz')
d_hae = np.load(ddir+hae_name+'.npz')
d_pod = np.load(ddir+pod_name+'.npz')

c_vae = d_vae["c"]
c_ae = d_ae["c"]
c_hae = d_hae["c"]
c_pod = d_pod["vh"][:r].T

#%%
print(c_vae.shape)
print(c_pod.shape)
print(c_ae.shape)
print(c_hae.shape)
print(c_hae.dtype)
print(np.unique(np.isnan(c_hae)))
#%%
coefs = np.zeros((4, r, r))

coefs[0,:,:] = np.corrcoef(c_hae.T,)
coefs[1,:,:] = np.corrcoef(c_ae.T)
coefs[2,:,:] = np.corrcoef(c_vae.T)
coefs[3,:,:] = np.corrcoef(c_pod.T)

#%%
detR = np.zeros(4)

for i in range(4):
    detR[i] = np.linalg.det(coefs[i])

detR = np.round(detR * 100, 2)

coefs = np.abs(coefs)

my_cmap = sns.color_palette('cmo.tempo', as_cmap=True)

fig, ax = plt.subplots(1, 4, sharey = True, figsize = (12, 4))
ax[0].imshow(coefs[0], cmap = my_cmap, vmin = 0, vmax = 1)
ax[1].imshow(coefs[1], cmap = my_cmap, vmin = 0, vmax = 1)
ax[2].imshow(coefs[2], cmap = my_cmap, vmin = 0, vmax = 1)
con = ax[3].imshow(coefs[3], cmap = my_cmap, vmin = 0, vmax = 1)

cax = fig.add_axes([ax[3].get_position().x1+0.02,ax[3].get_position().y0,0.02,ax[3].get_position().height])

cbar = fig.colorbar(con, cax=cax)
cbar.ax.locator_params(nbins = 5)

for axx in ax:
    axx.set_aspect('equal')
    axx.set_xticks(np.arange(0, 5))
    axx.set_xticklabels(np.arange(1, 6))
    axx.set_yticks(np.arange(0, 5))
    axx.set_yticklabels(np.arange(1, 6))
    axx.set_xlabel('$r_i$')

ax[0].set_ylabel('$r_i$')
ax[0].set_title(f'CNN--HAE--5 ({detR[0]})')
ax[1].set_title(f'CNN--AE--5 ({detR[1]})')
ax[2].set_title(f'CNN--$\\beta$VAE--5 ({detR[2]})')
ax[3].set_title(f'POD--5 ({detR[3]})')

fdir = './fig/'
plt.savefig(fdir + '4corrcoef5.png', bbox_inches='tight', dpi = 300)
# %%
