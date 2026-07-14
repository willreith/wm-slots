# Description of first pass analysis

## Goal
Carry out preliminary analysis to understand whether this is a worthwhile project to continue with.

## Approach
We want to replicate the working memory signatures from Stroud et al. (2023) to understand whether representations of objects and/or positions show the same signature of distinct encoding in early compared to mid/late delay periods. 

To do so, we will take trials in which only one object is shown to the animal. On these trials, we will either marginalise over the position or the object identity. We will then try to decode, respectively, the object identity or the position using linear discriminant analysis. The decoder will be trained on 100ms bins from the stimulus presentation (1000ms) until the end of the delay period (1000ms). Each decoder will be trained on 75% of trials and will be used to decode identity or position from held-out data (25%) from all time points with 4-fold cross-validation. 

In addition to doing this decoding analysis, we will also do representational similarity analysis (using cosine similarity) with the same binning and train/test strategy as for the decoding analysis.

## Methods detail

### Trials, marginalisation, and binning
We use only single-object trials, where one object of identity {a, b, c} appears at one of three
screen positions {0, 1, 2} (3×3 design). To decode **identity** we marginalise over position (pool
all positions and ask which object was shown); to decode **position** we marginalise over identity.
Each is therefore a 3-class problem, which pools trials across the other factor and so has more
trials per class (more power) than a joint 9-way identity×position decode.

Spikes are aligned to stimulus onset and binned into a rate tensor spanning the 2.0 s analysis
window (1.0 s stimulus + 1.0 s delay). The default bin is 50 ms (40 bins), giving a 40×40
cross-temporal matrix; the bin size is configurable (e.g. 100 ms → 20 bins).

### Why a pseudopopulation
Units are not simultaneously recorded across sessions, and even within a session not every unit is
observed on every trial: each unit carries a per-trial **observation mask**, and a zero spike count
in an *unobserved* trial is missing data, not silence (median coverage ≈0.67; only ~30% of units are
observed on all trials). A real population vector for a given trial therefore does not exist for most
units. A *pseudopopulation* sidesteps this: because we only need each class's *distribution* of
population responses (not the true simultaneous activity), we synthesise population vectors by
drawing, independently per unit, from that unit's own trials of the required class.

Construction (per cross-validation fold, per class): for each unit, sample with replacement a whole
real trial of that class from the unit's *observed* trials, and copy that trial's entire 20/40-bin
firing-rate trajectory into the pseudo-trial. Sampling a whole trial (rather than per-bin) keeps each
unit's temporal trajectory internally coherent, which is essential for the cross-temporal analysis.
Units unobserved for a class in that fold contribute zeros. Stacking many such pseudo-trials gives a
(pseudo-trials × units × bins) tensor with balanced classes. Units are z-scored per unit using the
training set's mean/SD so that no unit dominates on firing-rate scale alone.

### Linear discriminant analysis (LDA) and cross-temporal generalisation
LDA finds the linear projection of the population vector that best separates the class means relative
to within-class scatter; a test vector is assigned to the nearest class in that discriminant space.
We use **shrinkage LDA** (Ledoit–Wolf, `solver='lsqr', shrinkage='auto'`), which regularises the
covariance estimate — necessary because we have many units relative to trials.

The key manipulation is **cross-temporal**: we train one LDA on each *training* time bin and then
test it at *every* time bin, filling a (train-bin × test-bin) accuracy matrix (chance = 1/3). The
off-diagonal structure is the scientific readout:

- A **stable / gain code** uses a fixed coding axis over time, so a decoder trained early generalises
  to late bins → the matrix is a filled square (high off-diagonal).
- A **rotating / dynamic-slot code** moves the coding subspace through the delay, so a decoder trained
  early fails on late bins → accuracy concentrates in a band along the diagonal (off-diagonal
  collapse). This is the Stroud et al. (2023) dynamic-coding signature.

Training uses 75% of trials, testing the held-out 25%, with 4-fold cross-validation; the whole
procedure is repeated over many random pseudopopulation draws and averaged.

### Representational similarity analysis (RSA)
As a decoder-free complement we compute, for each time bin, a **class-coding vector** per class (the
class-mean population vector minus the grand mean over classes) and take the **cosine similarity**
between coding vectors at every pair of time bins (mean over classes). To avoid a spurious diagonal
of 1.0 from correlated noise, the two bins in each comparison come from independent
cross-validation folds. High off-diagonal cosine = the coding geometry is preserved across time
(stable); off-diagonal decay = the coding subspace rotates (dynamic). Same binning and train/test
strategy as the decoder.

### Aggregating over sessions
Sessions can be selected by criteria — minimum MUA units and good units *per region*, minimum
successful one-object trials, subject, and 3-position vs >3-position design — and the cross-temporal
matrices aggregated two ways:

- **average** (default): run the full decode/RSA within each session (its own pseudopopulation) and
  mean the resulting matrices, treating each session as an independent replicate (supports mean ± SEM
  across sessions).
- **pool**: concatenate units across sessions into a single large pseudopopulation and decode once,
  for maximum power. This assumes the three positions (and three identities) are physically
  comparable across sessions, which holds for the fixed 3-position design; pseudo-trials are paired
  randomly across sessions within each class.

Position decoding is restricted to the discrete **3-position** sessions (Perle 2022-05-26…06-05,
Elgar 2022-08-19…09-05); later sessions use ~continuous positions and are out of scope for position.
Sessions are loaded from disk when available, otherwise streamed from the DANDI archive.

