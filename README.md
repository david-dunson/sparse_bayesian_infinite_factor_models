# Sparse Bayesian Infinite Factor Models

We focus on sparse modelling of high-dimensional covariance matrices using Bayesian latent
factor models. We propose a multiplicative gamma process shrinkage prior on the factor loadings
which allows introduction of infinitely many factors, with the loadings increasingly shrunk
towards zero as the column index increases. We use our prior on a parameter-expanded loading
matrix to avoid the order dependence typical in factor analysis models and develop an efficient
Gibbs sampler that scales well as data dimensionality increases. The gain in efficiency is achieved
by the joint conjugacy property of the proposed prior, which allows block updating of the loadings
matrix. We propose an adaptive Gibbs sampler for automatically truncating the infinite loading
matrix through selection of the number of important factors. Theoretical results are provided
on the support of the prior and truncation approximation bounds. A fast algorithm is proposed
to produce approximate Bayes estimates. Latent factor regression methods are developed for
prediction and variable selection in applications with high-dimensional correlated predictors.
Operating characteristics are assessed through simulation studies, and the approach is applied to
predict survival times from gene expression data.

A. Bhattacharya and D. B. Dunson (2011). Sparse Bayesian Infinite Factor Models. *Biometrika* 98(2), pp. 291–306

`gendat.m`  generates simulated data

`spfactcovest_mgploadings.m` implements the Gibbs sampler in bka paper
