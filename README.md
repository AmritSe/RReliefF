# RReliefF
## Python Implementation of RReliefF - a feature selection tool for regression problems

RReliefF is a a feature selection tool for regression problems that helps in determining the predictive performance of the different features in a data set.

In addition to RReliefF, implementations of Relief and ReliefF - feature selection algorithms for classification problems, are also available in [relieff.py](relieff.py)

Although the function is python based, the function interface is designed to mimic the [Matlab implementation](https://www.mathworks.com/help/stats/relieff.html). 

## Implementation of Relief based algorithms
This code follows the algorithm for the Relief based algorithms as described in "An adaptation of Relief for attribute estimation in regression" by M. Robnik-Sikonja and I. Kononenko

Equation References used in the comments of the [main file](relieff.py) are based on the aforementioned article

To work with RReliefF specifically, use `W = RReliefF(X, y, opt)` 

`opt` can be replaced with the following optional arguments:

- `updates` - This can be 'all' (default) or a positive integer depending 
- `k` - The number of neighbours to look at. Default is 10.
- `sigma` - Distance scaling factor. Default is 50.
- `weight_track` - Returns a matrix which tracks the weight changes at each iteration. False by default

## Examples
Examples involving implementations for the 3 primary Relief based algorithms are included in [examples.py](examples.py). The variable `regressionProblem` can be set to `True` for trying out RReliefF and `False` for Relief and ReliefF. The examples included deliberately include a feature with random values to demonstrate how the Relief based algorithms can weed out irrelevant features in both, regression and classification problems.

RReliefF was also used in the following publication I co-authored (and serves as a good example of the algorithm's efficacy):

[A Comparative Study of Wavelet-based Descriptors for Fault Diagnosis of Self-Humidified Proton Exchange Membrane Fuel Cells](https://onlinelibrary.wiley.com/doi/full/10.1002/fuce.201900125)





