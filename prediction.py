#%%
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models
import keras.backend as K
from postprocessing.error import err


model_name = "AE_ld5"
model_save_name = "AE_ld5"
model_decoder_dir = "../models/de_{}.h5".format(model_name)
model_encoder_dir = "../models/en_{}.h5".format(model_name)
#%%

latent_dim = 5
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon

encoder = models.load_model(model_encoder_dir)
print(encoder.summary())

decoder = models.load_model(model_decoder_dir)
print(decoder.summary())

inp = encoder.layers[0].input
out_d = decoder(encoder(inp))

model = models.Model(inp, out_d,name=model_name)
print(model.summary())
#%%
d = np.load("../data/U_train.npz")
u = d["U"][:,:96,4:196]
u = u -u.mean(0)
print(u.shape)
u = u.reshape(-1,96,192)
u = np.expand_dims(u,-1)
print(u.shape)
print(f"INFO: Going to predict , the data has shape of{u.shape}")

if model_name[0] == "A":
    z = encoder.predict(u)
    u_p = model.predict(u)
elif model_name[0] == "V":
    z = encoder.predict(u)
    z = sampling([z[0],z[1]])
    u_p = decoder(z)
#%%
modes = decoder.predict(np.diag(np.ones(latent_dim))).squeeze()
e = err(u, u_p)
print(e)
#%%
plt.imshow(u[0, :, :])
plt.figure()
plt.imshow(u_p[0, :, :])
#%%
plt.figure()
plt.imshow(modes[0, :, :])
np.savez_compressed("/postprocessing/pred_data/"+model_save_name+".npz", u_p = u_p, modes = modes, c = z)
plt.show()
# %%
