# Phase Contrast Imaging Simulation

This repository contains a Python implementation of phase contrast microscopy simulation using the Transport of Intensity Equation (TIE) for phase retrieval.

## Overview

Phase contrast microscopy is a powerful technique for visualizing transparent specimens (like biological cells) that do not absorb light but alter its phase. This simulation demonstrates:

1. Generation of transparent phase objects
2. Wave propagation using the angular spectrum method
3. Phase retrieval using the Transport of Intensity Equation (TIE)
4. Phase contrast microscopy simulation

## How It Works

### Phase Object Generation
- Creates synthetic phase objects with Gaussian profiles
- Transparent objects that only modify the phase of light

### Wave Propagation
- Simulates light wave propagation using angular spectrum method
- Calculates defocused images at different z-planes

### Transport of Intensity Equation (TIE)
- Uses intensity measurements at different focal planes to recover phase information
- Solves TIE using Fourier methods

### Phase Contrast Microscopy
- Simulates the effect of a phase plate in a phase contrast microscope
- Converts invisible phase variations into visible intensity contrast

## Requirements
- numpy
- matplotlib
- scipy
- scikit-image

## Usage

```python
python phase_contrast_simulation.py
```

The simulation will:
1. Generate synthetic phase objects
2. Calculate intensity images at different focal planes
3. Reconstruct the phase using TIE
4. Simulate phase contrast microscopy imaging
5. Display comparison of bright-field and phase contrast images

## Results

The code generates visualization of:
- Original phase object
- Defocused images (front, in-focus, and back focal planes)
- Reconstructed phase using TIE
- Phase contrast microscope simulation

Comparing the bright-field image (where transparent objects are invisible) with the phase contrast image demonstrates how phase contrast microscopy makes transparent objects visible.

## Theory

### Transport of Intensity Equation

The TIE relates the intensity gradient along the optical axis to the phase distribution:

∇⊥ · [I(x,y,0) ∇⊥φ(x,y)] = -k ∂I(x,y,z)/∂z|z=0

Where:
- I is the intensity
- φ is the phase
- k is the wave number (2π/λ)
- ∇⊥ is the gradient operator in the transverse plane

### Phase Contrast Microscopy

Phase contrast microscopy works by:
1. Separating the direct (unscattered) and diffracted light
2. Advancing/retarding the phase of the direct light by π/2
3. Recombining the light to create amplitude contrast from phase differences

## References

- Teague, M. R. (1983). Deterministic phase retrieval: a Green's function solution. JOSA, 73(11), 1434-1441.
- Paganin, D., & Nugent, K. A. (1998). Noninterferometric phase imaging with partially coherent light. Physical review letters, 80(12), 2586.
- Zernike, F. (1942). Phase contrast, a new method for the microscopic observation of transparent objects. Physica, 9(7), 686-698.