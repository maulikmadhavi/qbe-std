import numpy as np
import matplotlib.pyplot as plt



def get_20hz_sine_wave(perturbation=0, dims=1):
    # Add noise to the sine wave if perturbation is not zero
    # perturbation = min(perturbation, 0.1)
    t = np.linspace(0, 1, 100)  # Time vector
    freq = 50  # Frequency in Hz
    sine_wave = np.expand_dims(np.sin(2 * np.pi * freq * t), 0).T   # [len(t), 1]
    noise = np.random.normal(0, perturbation, dims*len(t)).reshape(len(t), dims)  # [len(t), dims]
    return sine_wave + noise, t

def get_10hz_sine_wave(perturbation=0):
    # Add noise to the sine wave if perturbation is not zero
    perturbation = min(perturbation, 0.1)
    t = np.linspace(0, 1, 1000)  # Time vector
    freq = 10  # Frequency in Hz
    sine_wave = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, perturbation, len(t))
    return sine_wave + noise, t    

# plt.figure()
# plt.plot(t, sine_wave)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title(f'Sine Wave at {freq} Hz')
# plt.show()


