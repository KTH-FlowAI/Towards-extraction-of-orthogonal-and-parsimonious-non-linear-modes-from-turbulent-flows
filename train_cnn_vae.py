#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras import backend as K
from config.train_config import config
from scipy.io import savemat
from matplotlib import pyplot as plt

dirc = 'data/'

wdir ="models/" # path to save model
d = np.load(dirc + 'U_train.npz')
u = d["U"]
u = np.concatenate((u,u[0:1,:,:]),axis=0)
u = u[:,:96,4:196]
u_mean = u.mean(axis = 0)
u -= u_mean
u = np.expand_dims(u,-1)
print(u.shape)
nt, nx, ny, nv = u.shape


# Shuffle and Split train & validation dataset
ind_val = np.zeros(nt, dtype = bool)
n_val = int(nt * 0.2)
ind_val[:n_val] = True
np.random.seed(24)
np.random.shuffle(ind_val)

u_trn = u[~ind_val]
u_tst = u[ind_val]

print(u_trn.shape)
print(u_tst.shape)
latent_dim = config.latent_dim
beta = config.beta

name = f'VAE_ld{latent_dim}_b{beta}'
model_name = f'_{name}.h5'

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon

act = config.act
fs = config.knsize
ffs = config.poolsize
st = config.strides
inp = layers.Input(shape = (nx, ny, nv))
x = layers.Conv2D(16, fs, activation = act, strides = st, padding='same')(inp)
x = layers.MaxPooling2D(ffs, padding = 'same')(x)
x = layers.Conv2D(32, fs, activation = act, strides = st, padding='same')(x)
x = layers.MaxPooling2D(ffs, padding = 'same')(x)
x = layers.Conv2D(64, fs, activation = act, strides = st, padding='same')(x)
x = layers.MaxPooling2D(ffs, padding = 'same')(x)
x = layers.Conv2D(128, fs, activation = act, strides = st, padding='same')(x)
x = layers.MaxPooling2D(ffs, padding = 'same')(x)
x = layers.Conv2D(256, fs, activation = act, strides = st, padding='same')(x)
x = layers.MaxPooling2D(ffs, padding = 'same')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation=act)(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)
z = layers.Lambda(sampling)([z_mean, z_log_sigma])

encoder = models.Model(inp, [z_mean, z_log_sigma, z])
print(encoder.summary())

code_i = layers.Input(shape = (latent_dim,))
x = layers.Dense(128, activation=act)(code_i)
x = layers.Dense(4608, activation = act)(x)
x = layers.Reshape((3, 6, 256))(x)
x = layers.UpSampling2D(ffs)(x)
x = layers.Conv2D(256, fs, activation = act, strides = st, padding='same')(x)
x = layers.UpSampling2D(ffs)(x)
x = layers.Conv2D(128, fs, activation = act, strides = st, padding='same')(x)
x = layers.UpSampling2D(ffs)(x)
x = layers.Conv2D(64, fs, activation = act, strides = st, padding='same')(x)
x = layers.UpSampling2D(ffs)(x)
x = layers.Conv2D(32, fs, activation = act, strides = st, padding='same')(x)
x = layers.UpSampling2D(ffs)(x)
x = layers.Conv2D(16, fs, activation = act, strides = st, padding='same')(x)
out = layers.Conv2D(nv, fs, strides = st, padding='same')(x)

decoder = models.Model(code_i, out)
print(decoder.summary())

out_d = decoder(encoder(inp)[2])

model = models.Model(inp, out_d)
print(model.summary())

rec_loss = losses.mse(K.reshape(inp, (-1,)), K.reshape(out_d, (-1,)))
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(rec_loss + beta * kl_loss)
model.add_loss(vae_loss)
model.add_metric(rec_loss, name='rec_loss', aggregation='mean')
model.add_metric(kl_loss, name='kl_loss', aggregation='mean')

step = tf.Variable(0, trainable=False)
boundaries = [500, 100, 100]
values = [1e-3, 1e-4, 1e-5, 1e-6]
lr = optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

opt = optimizers.Adam(learning_rate = lr(step))
model.compile(optimizer = opt, loss = 'mse')

Epoch = config.epochs
# If unspecified, batch_size will default to 32
hist = model.fit(u_trn, u_trn, epochs = Epoch, validation_data = (u_tst, u_tst), verbose = 2,batch_size = 16)
savemat(wdir + f'loss_{name}.mat', hist.history)

encoder.save(wdir + 'en' + model_name)
decoder.save(wdir + 'de' + model_name)
