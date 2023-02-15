import numpy as np
from tensorflow.keras import models
from tensorflow.keras import backend as K


wdir = './'

name = 'VAE'
model_name = f'_{name}.h5'

latent_dim = 10
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon
#%%
encoder = models.load_model(wdir + 'en' + model_name, custom_objects={'sampling': sampling})
print(encoder.summary())

decoder = models.load_model(wdir + 'de' + model_name)
print(decoder.summary())

inp = encoder.layers[0].input
out_d = decoder(encoder(inp)[2])

model = models.Model(inp, out_d)
print(model.summary())


