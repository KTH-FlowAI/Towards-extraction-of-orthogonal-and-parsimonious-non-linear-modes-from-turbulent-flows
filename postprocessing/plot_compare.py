import numpy as np
from mat73 import loadmat
from matplotlib import pyplot as plt
from error import err

# plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 14)
plt.rc('legend', fontsize = 12, handletextpad=0.3)              
plt.rc('xtick', labelsize = 14)             
plt.rc('ytick', labelsize = 14)


dirc = 'data/'

d = np.load("../data/U_train.npz")
u = d['U'][:, :96, 4:196]

u_mean = u.mean(axis = 0)
u -=u_mean
nt, nx, ny = u.shape

ddir = 'pred_data/'
order_vae = np.loadtxt(ddir+'ordering_VAE_ld5_b1e-3.txt').astype(int)
order_ae = np.loadtxt(ddir+'ordering_AE_ld5.txt').astype(int)
d_vae = np.load(ddir+'VAE_ld5_b1e-3.npz')
d_ae = np.load(ddir+'AE_ld5.npz')
d_hae = np.load(ddir+'HAE_5.npz')
d_pod = np.load(ddir+'POD-5.npz')
u_hae = d_hae['u_p'].squeeze()
u_ae = d_ae['u_p'].squeeze()
u_vae = d_vae['u_p'].squeeze()
u_pod = d_pod['u_p'].squeeze()

u_pod = u_pod[:,:96,4:196]
print(u_hae.shape)
print(u_pod.shape)
print(u_ae.shape)
print(u_vae.shape)
x = np.linspace(-1, 5, 100)
y = np.linspace(-1.5, 1.5, 200)
y, x = np.meshgrid(y, x)
x = x[:96, :192]
y = y[:96, :192]
#%%
xb = np.array([-0.25, -0.25, 0.25, 0.25, -0.25])
yb = np.array([-0.25, 0.25, 0.25, -0.25, -0.25])

fig , ax = plt.subplots(5, 3, sharex = True, sharey = True, figsize = (10, 10))
plt.set_cmap('RdBu_r')
m = [0, 499, -1]
n = [1, 1000, 2000]
methods = ['HAE-5', 'AE-5', 'VAE-5', 'POD-5', 'Reference']
var = 1
for i in range(3):
    vmin = u[m[i], :, :].min()
    vmax = u[m[i], :, :].max()
    ax[0, i].contourf(x, y, u_hae[m[i], :, :], vmin = vmin, vmax = vmax, levels = 30)
    ax[1, i].contourf(x, y, u_ae[m[i], :, :], vmin = vmin, vmax = vmax, levels = 30)
    ax[2, i].contourf(x, y, u_vae[m[i], :, :], vmin = vmin, vmax = vmax, levels = 30)
    ax[3, i].contourf(x, y, u_pod[m[i], :, :], vmin = vmin, vmax = vmax, levels = 30)
    ax[4, i].contourf(x, y, u[m[i], :, :], vmin = vmin, vmax = vmax, levels = 30)
    #ax[i, 0].set_ylabel('$y$')
    ax[4, i].set_xlabel('$x$')
    ax[0, i].set_title(f'$n = {n[i]}$')
    
for i in range(5):
    if i == 0:
        ax[i, 1].set_title(methods[i] + f'  $n = {n[1]}$')
    else:
        ax[i, 1].set_title(methods[i])

# ax[5,1].set_title("Ground Truth")
for axx in ax.flatten():
    axx.set_aspect('equal')
    axx.fill(xb, yb, c = 'w')
    axx.fill(xb + 1.5, yb, c = 'w')
    axx.plot(xb, yb, c = 'k', lw = 1, zorder = 5)
    axx.plot(xb + 1.5, yb, c = 'k', lw = 1, zorder = 5)
plt.tight_layout()

plt.savefig("fig/"+'CNNAEs_rec.pdf', bbox_inches='tight')
