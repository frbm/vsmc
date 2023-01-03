# VSMC

## Directories:

```dmm_training```: where the training history for the Deep Markov model (DMM) is saved.

```training```: where the training history for the Linear Gaussian State-space (LGSSM) model is saved.

## Files:
```IWAE.py```: IWAE implementation.

```IWAE_Linear.py```: compare IWAE and VSMC on LGSSM.

```deep_markov.py```: model definition and training script for the DMM.

```linear_gaussian.py```: defines a class for LGSSM as a subclass of a VSMC model, training code at the end of the file.

```test_deep_markov.py```: unit tests on the DMM model architecture.

```vsmc.py```: defines a VSMC class.
