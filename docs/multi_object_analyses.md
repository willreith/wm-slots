**Regression-based subsapce identification**

Any multi-object trial can be represented by a 2-to-3-hot vector encoding object identities and positions.

We will take two approaches to define subspaces for encoding objects identites/positions in this task.

The first approach will use a compositional grammar to extract coding subspaces from two-object trial combinations, c={(0,1), (0,2), (1,2)}. We will compute regression weights for each neuron for each of the possible combinations of two positions. To get an estimate of the regression weight for a position, $\beta$(p), where p={0,1,2}, we first estimate the regression coefficient for all combinations, $\beta$(c). To obtain $\beta$(c), we add the combination coefficients containing p and subtract the combination coefficient that does not contain p. For example, if we wanted to determine the coefficients for the subspace representing items at position 1, we would get regressors for trials with items on the (0,1), (0,2) and (1,2) positions. We would then add the regressors for combinations containing position 1 and subtract the regressor that does not contain position 1 (and divide by 2):

$\beta$(1) = 1/2($\beta$(0,1) + $\beta$(1,2) - $\beta$(0,2)).

For each neuron, we will randomly split the trials in half 50 times, and for each split, we will compute the linear regression separately on each half. We will also apply a Lasso regularisation term to avoid overfitting, and select the regularisation amplitude with the highest maximum likelihood. The regression coefficient for a given combination will be the average of the 100 estimates.

Once we have obtained the regression coefficents for each position, we will use them to estimate the low-dimensional subspaces that capture the most task variance. With N neurons, this will give us an N-dimensional vector, **$\beta$**(p) = [$\beta$<sub>1</sub>, ..., $\beta$<sub>N</sub>]<sup>T</sup> that represents the activities of neurons that define the subspace for position p.

This approach marginalises over object identities. To identify identity-based subspaces instead of position-based subspaces, we would marginalise over positions instead.

The second approach will use a similar regression-based analysis on 3-object trials. We will table this for now.
