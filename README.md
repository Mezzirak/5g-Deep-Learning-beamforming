# 5g-beamforming-project

JUST STARTED 09-09-2025, SO THIS IS AT THE ABSOLUTE BEGINNING RIGHT NOW
ALL OF BELOW IS WHAT IS I HOPE TO EXECUTE IN THIS PROJECT - NOT WHAT IS ALREADY DONE.

I am intending for this project to one day to design, simulate and evaluate a deep learning-based beamformer for adaptive noise cancellation in a 5G multi-user wireless environment. The performance of the neural network model will be benchmarked against a classical Minimum Variance Distortionless Response (MVDR) beamformer.

The core objective is to mitigate two primary forms of interference in modern wireless communications:

1.  **Additive White Gaussian Noise (AWGN):** Fundamental noise inherent in electronic systems.
2.  **Inter-User Interference (IUI):** Signal corruption caused by other users transmitting on the same frequency in a multi-user scenario.

The simulation will compare the effectiveness of a classical Digital Signal Processing (DSP) algorithm (MVDR) against a deep learning approach in maximising the Signal-to-Interference-plus-Noise Ratio (SINR) for a desired user.

**BACKGROUND THEORY**

If you have one base station (known as a gNB in 5G terminology) and multiple users (User Equipment or UEs). The gNB has an array of antennas, which is what allows it to perform beamforming. The signal received by the gNB's antenna array is a superposition of the signals from all users, plus noise. 

The project simulates a multi-user wireless environment. The signal, y(t), received by the base station's multi-antenna array is modelled as a linear combination of signals from all active users, corrupted by additive noise.

The mathematical model is represented as:

$$
\mathbf{y}(t) = \sum_{k=1}^{K} \mathbf{h}_k s_k(t) + \mathbf{n}(t)
$$

Where:
- $K$ is the total number of users
- $\mathbf{y}(t)$ is the **received signal vector** at the base station's antenna array
- $s_k(t)$ is the **signal transmitted by the k-th user**
- $\mathbf{h}_k$ is the **channel vector** that characterises the path (including fading and phase shifts) between the k-th user and the antenna array
- $\mathbf{n}(t)$ is the **Additive White Gaussian Noise (AWGN) vector**

**Learning resouces**

- Machine Learning for Signal Processing, Max A. Little
- Signal Processing for Communications Paolo Prandoni & Martin Vetterli
- A Brief Introduction to Machine Learning for Engineers by O. Simeone (Foundations and Trends in Signal Processing, 2018)
- Deep Learning by Goodfellow, Bengio & Courville (2016)
- Reinforcement Learning: An Introduction by Sutton & Barto (2018 ed.)
- Pattern Recognition and Machine Learning by Bishop (2006)
