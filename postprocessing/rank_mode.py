import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import backend as K
from error import err


plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1.5)
plt.rc('font', size = 14)
plt.rc('legend', fontsize = 12, handletextpad=0.3)              
plt.rc('xtick', labelsize = 14)             
plt.rc('ytick', labelsize = 14)


latent_dim =5
# model_name = "AE_ld5"
model_name = "VAE_ld5_b0.001"
# model_save_name = "AE_ld5"
model_save_name = "VAE_ld5_b1e-3"
model_decoder_dir = "../models/de_{}.h5".format(model_name)


decoder = models.load_model(model_decoder_dir)
d = np.load("../data/U_train.npz")
u = d["U"][:,:96,4:196]
u = u -u.mean(0)
print(u.shape)
u = u.reshape(-1,96,192)
u = np.expand_dims(u,-1)
nt, nx, ny, nv = u.shape


data = np.load("pred_data/"+model_save_name+".npz")
u_p = data['u_p']
c = data['c']

et = err(u, u_p)

m = np.zeros((latent_dim), dtype=int)
n = np.arange(latent_dim)
e = np.zeros((latent_dim))

for i in range(latent_dim):
    Eks = []
    for j in n:
        print(m[:i], j)
        c_z = np.zeros(c.shape)
        c_z[:, m[:i]] = c[:, m[:i]]
        c_z[:, j] = c[:, j]
        u_pz = decoder.predict(c_z)
        Eks.append(err(u, u_pz))
    Eks = np.array(Eks).squeeze()
    print(Eks)
    ind = n[np.argmax(Eks)]
    m[i] = ind
    n = np.delete(n, np.argmax(Eks))
    
    e[i] = np.max(Eks)
    print(e[i], ind)
    

np.savetxt(f'pred_data/ordering_{model_save_name}.txt', m)

