# 5g-beamforming-project

This project is in progress and will design, simulate and evaluate a deep learning-based beamformer for adaptive noise cancellation in a 5G multi-user wireless environment. The performance of the neural network model will be benchmarked against a classical Minimum Variance Distortionless Response (MVDR) beamformer.

The core objective is to mitigate two primary forms of interference in modern wireless communications:

1.  **Additive White Gaussian Noise (AWGN):** Fundamental noise inherent in electronic systems.
2.  **Inter-User Interference (IUI):** Signal corruption caused by other users transmitting on the same frequency in a multi-user scenario.

The simulation will compare the effectiveness of a classical Digital Signal Processing (DSP) algorithm (MVDR) against a deep learning approach in maximising the Signal-to-Interference-plus-Noise Ratio (SINR) for a desired user.

## Background Theory

### The Challenge of Multi-User Interference

In modern 5G wireless systems, a single base station (gNB) must communicate simultaneously with multiple user devices (UEs). The gNB uses an array of antennas to receive signals, but the signal that arrives at this array is a superposition, a mixture of the signals from all users, corrupted by inherent electronic noise. This creates a significant challenge: how to isolate the weak signal of one desired user from the overwhelming interference of all other users transmitting at the same time.

The solution is adaptive beamforming, a signal processing technique that allows the antenna array to "listen" in a specific direction. By intelligently combining the signals received at each antenna, the gNB can form a highly focused beam towards the desired user, enhancing their signal while simultaneously suppressing interfering signals from other directions.

### The System Signal Model

To design a beamformer, we first model the received signal vector, $\mathbf{y}(t)$, at the gNB's antenna array as:

$$
\mathbf{y}(t) = \underbrace{\mathbf{h}_d s_d(t)}_{\text{Desired Signal}} + \underbrace{\sum_{k \neq d}^{K} \mathbf{h}_k s_k(t)}_{\text{Inter-User Interference}} + \underbrace{\mathbf{n}(t)}_{\text{Noise}}
$$

Where:
- $K$ is the total number of users
- $\mathbf{y}(t)$ is the **received signal vector**
- $s_k(t)$ is the **signal transmitted by the k-th user**
- $\mathbf{h}_k$ is the **channel vector** for the k-th user. This vector represents the unique spatial signature of the user's signal path to the antenna array
- $\mathbf{n}(t)$ is the **Additive White Gaussian Noise (AWGN) vector**

The goal of the beamformer is to process $\mathbf{y}(t)$ to recover an accurate estimate of the desired signal, $s_d(t)$

### The MVDR Beamformer: A Classical Solution

The Minimum Variance Distortionless Response (MVDR) beamformer is a widely used algorithm for this task. It calculates an optimal set of weights, $\mathbf{w}$, for the antenna array to minimise the power from interference and noise while maintaining a distortionless response (a gain of 1) in the direction of the desired user

The formula for the MVDR weights is:

$$
\mathbf{w}_{\text{mvdr}} = \frac{\mathbf{R}^{-1}\mathbf{a}(\theta_d)}{\mathbf{a}(\theta_d)^H \mathbf{R}^{-1}\mathbf{a}(\theta_d)}
$$

Where $\mathbf{R}$ is the covariance matrix of the received signal and $\mathbf{a}(\theta_d)$ is the steering vector of the desired user. This project uses the MVDR beamformer as the classical benchmark against which a novel deep learning approach is compared

**Project Sturcture**

main.py: Runs the complete project pipeline, from data generation to final evaluation

simulation.py: Generates the 5G multi-user signal data and simulates the wireless channel

models.py: Defines the architectures for both the classical MVDR beamformer and the deep learning model

evaluate.py: Contains functions to calculate key performance metrics like Bit Error Rate (BER)

plotting.py: Includes all functions for creating visualisations, such as constellation and beam pattern plots

**Learning resouces**

- Machine Learning for Signal Processing, Max A. Little
- Signal Processing for Communications Paolo Prandoni & Martin Vetterli
- A Brief Introduction to Machine Learning for Engineers by O. Simeone (Foundations and Trends in Signal Processing, 2018)
- Deep Learning by Goodfellow, Bengio & Courville (2016)
- Reinforcement Learning: An Introduction by Sutton & Barto (2018 ed.)
- Pattern Recognition and Machine Learning by Bishop (2006)
