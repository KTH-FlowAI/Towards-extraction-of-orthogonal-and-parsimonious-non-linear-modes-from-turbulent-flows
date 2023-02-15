import numpy as np 


d1 = np.load("U_pt1.npz")
d2 = np.load("U_pt2.npz")
U1 = d1["U"]
U2 = d2["U"]

print(U1.shape)
print(U2.shape)

U = np.concatenate([U1,U2],axis=0)
print(f"The dataset has shape of {U.shape}")
np.savez_compressed("U_train.npz",U=U)
