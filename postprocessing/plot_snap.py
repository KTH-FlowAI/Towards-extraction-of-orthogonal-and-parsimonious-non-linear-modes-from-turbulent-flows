import numpy as np
from matplotlib import pyplot as plt
from error import err
import seaborn as sns
cmp = sns.color_palette('YlGnBu_r', as_cmap=True)
plt.set_cmap(cmp)
"""
Plot one flow fleid snapshot for Beta-VAE, AE, HAE, POD and reference. 

The energy percentage E_k have been annotated for 4 methods respectively

Return:
    A .png file 
"""

d = np.load("../data/U_train.npz")
u = d['U']
u = u[:, :96, 4:196]
u = np.expand_dims(u,-1)
u_mean = u.mean(axis = 0)
u -= u_mean
nt, nx, ny,nv = u.shape
print(u.shape)

ddir = 'pred_data/'
d_vae = np.load(ddir+'VAE_ld5_b1e-3.npz')
d_ae = np.load(ddir+'AE_ld5.npz')
d_hae = np.load(ddir+'HAE_5.npz')
d_pod = np.load(ddir+'POD-5.npz')

u_hae = d_hae['u_p']
u_ae = d_ae['u_p']
u_vae = d_vae['u_p']
u_pod = d_pod['u_p']
u_pod = u_pod[:,:96,4:196]
u_pod = np.expand_dims(u_pod,-1)

print(u_hae.shape)
print(u_pod.shape)
print(u_ae.shape)
print(u_vae.shape)

u_ps = [u_vae,u_hae,u_ae,u_pod]
errors =[]
for u_p in u_ps:
    error = err(u[:,:,:,0],u_p[:,:,:,0])
    print(error)
    errors.append(np.round(error,3)*100)

indx = 710
u_plot = np.concatenate((u[indx,:,:,0:1],u_vae[indx,:,:,0:1],u_hae[indx,:,:,0:1],u_ae[indx,:,:,0:1],u_pod[indx,:,:,0:1]),axis=-1)
print(u_plot.shape)




#####################################
x = np.linspace(-1, 5, 100)
y = np.linspace(-1.5, 1.5, 200)
y, x = np.meshgrid(y, x)
x = x[:96, :192]
y = y[:96, :192]
xb = np.array([-0.25, -0.25, 0.25, 0.25, -0.25])
yb = np.array([-0.25, 0.25, 0.25, -0.25, -0.25])

fig , ax = plt.subplots(2, 3, sharex = True, sharey = True, figsize = (16, 10))
plt.set_cmap('RdBu_r')

methods = [ 'Reference', 'VAE-5','HAE-5', 'AE-5', 'POD-5']
var = 1
ax = ax.flatten()
ax[-1].axis("off")

for i in range(5):
    vmin = u_plot[:,:,0].min()
    vmax = u_plot[:,:,0].max()
    clb = ax[i].contourf(x, y, u_plot[:,:,i], vmin = vmin, vmax = vmax, levels = 30)
    #ax[i, 0].set_ylabel('$y$')
ax[-1].set_xlabel('$x$')
# ax[0].set_title(f'$n = {n[i]}$')    
for i in range(5):
    if i == 0:

        ax[i].set_title(methods[i],fontdict={"size":16})
    else:
        ax[i].set_title(methods[i]+f"({errors[i-1]}%)",fontdict={"size":16})
    
# ax[4].set_title("Ground Truth")
for axx in ax.flatten()[:-1]:
    axx.set_aspect('equal')
    axx.fill(xb, yb, c = 'w',zorder =3)
    axx.fill(xb + 1.5, yb, c = 'w',zorder=3)
    axx.plot(xb, yb, c = 'k', lw = 1, zorder = 5)
    axx.plot(xb + 1.5, yb, c = 'k', lw = 1, zorder = 5)
    axx.set_aspect('equal')
plt.tight_layout()
cbar = fig.colorbar(clb, ax=ax.flatten(),orientation="horizontal")
cbar.ax.locator_params(nbins = 4)
cbar.ax.tick_params(labelsize=18)
plt.savefig("fig/"+'Final_CNNAEs_re', bbox_inches='tight')
