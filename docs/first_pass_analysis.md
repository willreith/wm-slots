# Description of first pass analysis

## Project description
Gain model or dynamic slots? Reanalysing prefrontal working memory represent- ations with rotational dynamics
Working memory (WM) allows animals to store recent information for imminent use, yet the nature of WM representations in prefrontal cortex (PFC) remains debated. Watters et al. (2025, bioRxiv) presented rhesus macaques with multiple simultaneous objects, one of which subsequently determined a rewarded saccade. They compared several models of WM representations and found that PFC activity is consistent with a gain model: objects represented by weighted activity in a shared neural pool. This contrasts with a slot model, in which objects are represented in separate subspaces - an architecture that explains WM representations in sequence-WM tasks (Xie et al., 2022, Science; El-Gaby et al., 2024, Nature; Jensen et al., 2025, bioRxiv). The discrepancy in explanatory power across disparate WM tasks may reflect differing task demands, but another possibility is that Watters et al. only tested a limited version of the slot model. Stroud et al. (2023) showed that non-sequence WM representations rotate through a series of subspaces during delay periods and recent theoretical work suggests that such rotational dynamics are both energetically and computationally efficient (Dorrell et al., 2026, bioRxiv).
We propose to reanalyse the Watters et al. (2025) dataset to test whether PFC WM representations are better described by a gain model or a dynamic slot model incorporating rotational dynamics, a comparison their original analyses did not make.

## Data
https://doi.org/10.48324/dandi.000620/0.260127.2208

## Background reading
Dorrell, W., Latham, P. E., Behrens, T. E. J., & Whittington, J. C. R. (2026). An Efficient Computing Theory of Prefrontal Structured Working Memory Representations (p. 2026.02.16.706126). bioRxiv. https://doi.org/10.64898/2026.02.16.706126
Jensen, K. T., Doohan, P., Sablé-Meyer, M., Reinert, S., Baram, A., Akam, T., & Behrens, T. E. J. (2025). A mechanistic theory of planning in prefrontal cortex (p. 2025.09.23.677709). bioRxiv. https://doi.org/10.1101/2025.09.23.677709
Johnston, W. J., Fine, J. M., Yoo, S. B. M., Ebitz, R. B., & Hayden, B. Y. (2024). Semi-orthogonal subspaces for value mediate a binding and generalization trade-off. Nature Neuroscience, 1–13. https://doi.org/10.1038/s41593-024-01758-5
Stroud, J. P., Watanabe, K., Suzuki, T., Stokes, M. G., & Lengyel, M. (2023). Optimal information loading into working memory explains dynamic coding in the prefrontal cortex. Proceedings of the National Academy of Sciences, 120(48), e2307991120. (world). https://doi.org/10.1073/pnas.2307991120
Watters, N., Gabel, J., Tenenbaum, J., & Jazayeri, M. (2026). Working Memory of Multi-Object Scenes in Primate Frontal Cortex (p. 2026.01.27.702062). bioRxiv. https://doi.org/10.64898/2026.01.27.702062
Xie, Y., Hu, P., Li, J., Chen, J., Song, W., Wang, X.-J., Yang, T., Dehaene, S., Tang, S., Min, B., & Wang, L. (2022). Geometry of sequence working memory in macaque prefrontal cortex. Science, 375(6581), 632–639. https://doi.org/10.1126/science.abm0204

## Goal
Carry out preliminary analysis to understand whether this is a worthwhile project to continue with.

## Approach
We want to replicate the working memory signatures from Stroud et al. (2023) to understand whether representations of objects and/or positions show the same signature of distinct encoding in early compared to mid/late delay periods. 

To do so, we will take trials in which only one object is shown to the animal. On these trials, we will either marginalise over the position or the object identity. We will then try to decode, respectively, the object identity or the position using linear discriminant analysis. The decoder will be trained on 100ms bins from the stimulus presentation (1000ms) until the end of the delay period (1000ms). Each decoder will be trained on 75% of trials and will be used to decode identity or position from held-out data (25%) from all time points with 4-fold cross-validation. 

In addition to doing this decoding analysis, we will also do representational similarity analysis (using cosine similarity) with the same binning and train/test strategy as for the decoding analysis.

