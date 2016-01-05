# Restricted Boltzmann Machine

[![GoDoc](https://godoc.org/github.com/taylorchu/rbm?status.svg)](https://godoc.org/github.com/taylorchu/rbm)

[A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)


This package contain the following types of rbm:

- Binary (binary-binary, bernoulli-bernoulli)
- Gaussian (gaussian-binary, gaussian-bernoulli, grbm, gbrbm, real-valued)
- Classifier (softmax)

Both training and reconstruction should have zero allocation.
