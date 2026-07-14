

1. Define stim/early/late LDA subspace and project trajectories (Xie Fig3D)

We want to project activity onto LDA subspaces from decoders trained in the following time windows:
- 200-500ms
- 700-1000ms
- 1000-1300ms
- 1350-1650ms
- 1700-2000ms

To do so, we want to follow the decoder-based state space trajectory approach from Xie et al.
(Science, 2022). We will train decoders in each of these time windows and perform singular 
value decomposition on the projection matrices. This should yield two orthonormal right 
singular vectors of length N. We will then project the single-trial timeseries data 
(by position or identity) onto these subspace.

2. Confusion matrices for decoding on single-object error trials.

So far, we have only looked at decoding the position on correct trials. We would like to apply
this decoder on incorrect trials in order to figure out if the wrong position being encoded
in DMFC or FEF is what is leading to these errors.


3. PCA on single-object correct vs error trials.