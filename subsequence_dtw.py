"""Implementation of subsequence DTW algorithm.

There are three stages  
1. Compute the distance matrix
2. Compute the accumulated distance matrix
3. Compute the optimal path using back tacking algorithm
"""

from utils.gen_data import get_20hz_sine_wave, get_10hz_sine_wave
import numpy as np
import matplotlib.pyplot as plt
import librosa

def extract_mfcc(audio_file):
    audio, sample_rate = librosa.load(audio_file)
    x = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Normalize the MFCCs: mean = 0, std = 1
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)    
    return x

query = extract_mfcc("data/d124cd78-78ce-4603-8ffa-673f35887e61.wav")
reference = extract_mfcc("data/reference.wav")

def compute_distances_einsum(x, y):
    """
    Compute pairwise Euclidean distances between vectors in x and y using einsum.

    Args:
        x: Array of shape (Nx, D) containing Nx vectors of dimension D.
        y: Array of shape (Ny, D) containing Ny vectors of dimension D.

    Returns:
        distances: Array of shape (Nx, Ny) containing the Euclidean distances between each pair of vectors from x and y.
    """

    # Sum of squares for each vector in x and y
    sum_x = np.sum(x**2, axis=1)
    sum_y = np.sum(y**2, axis=1)

    # Compute dot products between x and y
    # x: [Nx, D], y: [Ny, D] -> xy: [Nx, Ny]
    xy = np.einsum('nd,md->nm', x, y)

    # Compute squared Euclidean distances
    # Use broadcasting to expand sum_x and sum_y to match the shape of xy
    distances_squared = sum_x[:, np.newaxis] + sum_y[np.newaxis, :] - 2 * xy

    return np.sqrt(np.maximum(distances_squared, 0))

print(query.shape, reference.shape)
d = compute_distances_einsum(query.T, reference.T)
print(d.shape)
lq = query.shape[1]
lr = reference.shape[1]

# Initialization
S = np.zeros((lq, lr))
T = np.zeros((lq, lr))
P = np.zeros((lq, lr))

# -> 1st row
print(S.shape)
S[0, :] = d[0, :]
T[0,:] = 0
P[0,:] = np.arange(lr)

# 1st column
S[1:, 0] = S[:lq-1,0] + d[1:, 0]
T[1:, 0] = np.arange(1, lq)
P[1:, 0] = 0


# Path tracking
for i in range(1, lq):
    for j in range(1, lr):
        res = np.argmin([S[i-1, j], S[i-1, j-1], S[i, j-1]])
        if res == 0:
            i_opt = i-1
            j_opt = j
        elif res == 1:
            i_opt = i-1
            j_opt = j-1
        else:
            i_opt = i
            j_opt = j-1
        S[i, j] = d[i, j] + S[i_opt, j_opt]
        T[i, j] = T[i_opt, j_opt] + 1
        P[i, j] = P[i_opt, j_opt] 
        
        
        
# Plot S matrix
plt.figure(figsize=(10, 5))
plt.imshow(S, cmap='jet')
plt.xlabel('Query sequence length')
plt.ylabel('Reference sequence length')
plt.title('S matrix')
plt.colorbar()
plt.show()