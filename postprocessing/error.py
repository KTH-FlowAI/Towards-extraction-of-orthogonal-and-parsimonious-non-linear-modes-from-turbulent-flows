import numpy as np

def err_norm(u, u_p):
    err = np.linalg.norm(u - u_p, axis = (1, 2))**2/np.linalg.norm(u, axis = (1, 2))**2
    return 1 - err.mean(axis = 0)


def err(u, u_p):
    err = np.sum((u - u_p)**2, axis = (1, 2))/np.sum(u**2, axis = (1, 2))
    return 1 - err.mean(axis = 0)

def mse(u, u_p):
    err = np.mean((u - u_p)**2, axis = (1, 2))
    return err.mean(axis = 0)