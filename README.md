<h1 align="center">
<p>AFLI :snowflake:</p>
<p align="center">
<img alt="GitHub" src="https://img.shields.io/github/license/cross-caps/AFLI?color=green&logo=GNU&logoColor=green">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?logo=python">
<img alt="colab" src="https://colab.research.google.com/assets/colab-badge.svg?">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.1.0-orange?logo=tensorflow">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>
<h2 align="center">
<p>Activation Function design: Linearity and effective Initialization</p>
</h2>


Code to reproduce plots from the paper: 
Activation function design for deep networks: linearity and effective initialization https://arxiv.org/abs/2105.07741


## Colab Notebooks

- For implementation of Variance and Covariance maps; see [Notebook](./Notebook/RTM_theory.ipynb)
- For analysing an activation function via correlation, moment ratio and dynamical isometry bounds; see [Notebook](./Notebook/Figure_Correlation_Moment_Ratio_Bounds.ipynb)


## Training and Testing

- For training a DNN model on MNIST/Fashion-MNIST/Cifar-10 dataset; see [script](./scripts/train.py)
  - require pre-computed values of <img src="https://latex.codecogs.com/gif.latex?(\sigma_w,\sigma_b)" title="(\sigma_w,\sigma_b)" /> for a given <img src="https://latex.codecogs.com/gif.latex?q^*" title="q^*" />; use functions from [Notebook](./Notebook/RTM_theory.ipynb) 
- For training with fixed value of parameter 'a' of an activation function; import [script](./scripts/utils_fixed_a.py)
  - *train* and *trainO* functions are provided to train with Gaussian or orthogonal weights 
- For training with variable value of parameter 'a' of an activation function; import [script](./scripts/utils_variable_a.py)
  - Only implementation with *htanh* activation is provided for demo purpose


## Contact 

Vinayak Abrol <abrol@maths.ox.ac.uk>

Michael Murray <murray@maths.ox.ac.uk>
