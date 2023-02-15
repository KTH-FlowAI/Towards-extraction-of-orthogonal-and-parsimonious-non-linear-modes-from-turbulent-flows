#%%
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import models
#%%
d = np.load("../data/U_train.npz")
u = d["U"]
u = u -u.mean(0)
u = u.reshape(-1,100,200,1)
u_mean = u.mean(axis = 0)
u -= u_mean
print(f"U shape: {u.shape}")
nt, nx, ny,nv= u.shape
print(f"The shape of u fluct is {nt,nx,ny}")
#%% POD
r = 5
u = u.reshape((nt, -1)).T
U, s, vh = np.linalg.svd(u,full_matrices=False)
print(U.shape)
# coefs = np.linalg.lstsq(U, u)[0]
# print(coefs.shape)
U = U[:, :r]
s = s[:r]
vh = vh[:r]
# coefs = coefs[:r]
u_p = U @ np.diag(s) @ vh
u = u.T.reshape((-1, nx, ny))
u_p = u_p.T.reshape((-1, nx, ny))
U = U.T.reshape((-1, nx, ny))
#%%
err = np.sum((u - u_p)**2, axis = (1, 2))/np.sum(u**2, axis = (1, 2))
err = (1-err.mean())*100
energy = np.round(err,2)
print(f"Use mode = {r}, the energy level is {energy}%")
#%%
np.savez_compressed( "./pred_data/"+f"POD-{r}"+".npz",
                    u_p = u_p,
                    modes = U,
                    vh = vh,
                    s = s)

u_plot = np.concatenate((u_p[1,:,:],u[1,:,:]),axis=-1)
#%%
model_name = f"POD-{r}"
x_grid  = np.linspace(-1,5,100)
z_grid  = np.linspace(-1.5,1.5,200)
x_grid,z_grid = np.meshgrid(x_grid,z_grid)

plt.figure()
xb = np.array([-0.25, -0.25, 0.25, 0.25, -0.25])
yb = np.array([-0.25, 0.25, 0.25, -0.25, -0.25])
fig,axx = plt.subplots(2,1,figsize=(16,9))
clb = axx[0].contourf(x_grid[4:196,:96].T,z_grid[4:196,:96].T,u_p[1,:96,4:196],cmap ="RdBu",vmax=u_plot.max(),vmin=u_plot.min())
axx[0].fill(xb, yb, c = 'w', zorder = 3)
axx[0].fill(xb + 1.5, yb, c = 'w', zorder = 3)
axx[0].set_xticks([0,2,4])
axx[0].set_yticks([-1,0,1])
axx[0].set_title("Pred ({}%)".format(energy),fontdict={"size":20})
axx[0].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
axx[0].plot(xb + 1.5, yb, c = 'k', lw = 1, zorder = 5)
axx[0].set_aspect("equal")


clb = plt.contourf(x_grid[4:196,:96].T,z_grid[4:196,:96].T,u[1,:96,4:196],cmap ="RdBu",vmax=u_plot.max(),vmin=u_plot.min())
axx[1].fill(xb, yb, c = 'w', zorder = 3)
axx[1].fill(xb + 1.5, yb, c = 'w', zorder = 3)
axx[1].set_xticks([0,2,4])
axx[1].set_yticks([-1,0,1])
axx[1].set_title("Ground Truth",fontdict={"size":20})
axx[1].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
axx[1].plot(xb + 1.5, yb, c = 'k', lw = 1, zorder = 5)
axx[1].set_aspect("equal")
# plt.colorbar(clb)
plt.tight_layout()
cbar = fig.colorbar(clb, ax=axx.flatten())
cbar.ax.locator_params(nbins = 5)
plt.savefig("./fig/"+model_name,bbox_inches="tight")
