import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

"""
Plot the ranked first 5 modes for Beta-VAE, AE, HAE and POD

Return:
    .png file
"""

plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 2)
plt.rc('font', size = 14)
plt.rc('legend', fontsize = 12, handletextpad=0.3)              
plt.rc('xtick', labelsize = 14)             
plt.rc('ytick', labelsize = 14)

x = np.linspace(-1, 5, 100)
y = np.linspace(-1.5, 1.5, 200)
y, x = np.meshgrid(y, x)
x = x[:96, 4:196]
y = y[:96, 4:196]



ddir = 'pred_data/'
order_vae = np.loadtxt(ddir+'ordering_VAE_ld5_b1e-3.txt').astype(int)
order_ae = np.loadtxt(ddir+'ordering_AE_ld5.txt').astype(int)
d_vae = np.load(ddir+'VAE_ld5_b1e-3.npz')
d_ae = np.load(ddir+'AE_ld5.npz')
d_hae = np.load(ddir+'HAE_5.npz')
d_pod = np.load(ddir+'POD-5.npz')

m_vae = d_vae['modes'][:, :, :]
m_vae = m_vae[order_vae]

m_ae = d_ae["modes"][:,:,:]
m_ae = m_ae[order_ae]

m_hae = d_hae["modes"][:,:,:].squeeze()

print(m_ae.shape)
print(m_vae.shape)
print(m_hae.shape)

m_pod = d_pod['modes'][:, :96, 4:196]

##############################3

modes = [m_vae, m_pod,m_ae,m_hae]
models = ['(a) CNN--$\\beta$VAE--5', '(b) POD--5', '(c) CNN--AE--5', '(d) CNN--HAE--5']


xb = np.array([-0.25, -0.25, 0.25, 0.25, -0.25])
yb = np.array([-0.25, 0.25, 0.25, -0.25, -0.25])

fig , ax = plt.subplots(5, 4, sharex = True, sharey = True, figsize = (12, 8))

cmp = sns.color_palette('cmo.curl', as_cmap=True)
cmp = sns.color_palette('cmo.curl', as_cmap=True)
plt.set_cmap(cmp)

for i in range(4):
    m = modes[i]
    axc = ax[:, i]
    q = 0
    for axx in axc.flatten():
        u = m[q, :, :]
        u = (u - u.mean())/np.std(u)
        axx.contourf(x, y, u, levels = 30,cmap ='RdBu')
        axx.set_aspect('equal')
        axx.fill(xb, yb, c = 'w', zorder = 3)
        axx.fill(xb + 1.5, yb, c = 'w', zorder = 3)
    
        axx.plot(xb, yb, c = 'k', lw = 1, zorder = 5)
        axx.plot(xb + 1.5, yb, c = 'k', lw = 1, zorder = 5)
        
        if i == 0:
            axx.set_ylabel('$z/h$')
        q += 1
    axc[0].set_title(models[i], fontsize = 24, pad = 15)
    axc[4].set_xlabel('$x/h$')

plt.tight_layout()

fdir = 'fig/'
plt.savefig(fdir + 'modes_5v5.png', bbox_inches='tight', dpi = 300)