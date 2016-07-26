---
layout: page
title: Keeping Up
subtitle: My list of research papers and blog posts on the most recent Data Science Best Practices
---

## Predictive Modeling

#### [Subsemble: An Ensemble Method](http://www.ncbi.nlm.nih.gov/pubmed/24778462)

Key Quote: 

> Subsemble partitions the full dataset into subsets of observations, fits a specified underlying algorithm on each subset, and uses a clever form of V-fold cross-validation to output a prediction function that combines the subset-specific fits. We give an oracle result that provides a theoretical performance guarantee for Subsemble.

#### [Confidence Intervals for Random Forests](http://jmlr.csail.mit.edu/papers/volume15/wager14a/wager14a.pdf)

Key Quote: 

> Estimating the variance of bagged learners based on the preexisting bootstrap replicates can be challenging, as there are two distinct sources of noise. In addition to the sampling noise (i.e., the noise arising from randomness during data collection), we also need to control the Monte Carlo noise arising from the use of a finite number of bootstrap replicates. We study the effects of both sampling noise and Monte Carlo noise.

The paper is a technical description of the topic but there is also an R package called [randomForestCI](https://github.com/swager/randomForestCI) which provides an easy way to generate confidence intervals from a `randomForest` object. The package does have a small bug when the predicted values are character values but hopefully the authors will fix that soon. 

#### [Impact Coding for High Cardinality Categorical Attributes](http://dl.acm.org/citation.cfm?id=507538)

Key Quote:

> This paper presents a simple data-preprocessing scheme that transforms high-cardinality categorical attributes into quasi-continuous scalar attributes suited for use in regression-type models. The key transformation used in the proposed scheme is one that maps each instance (value) of a high-cardinaltiy categorical to the probability estimate of the target attribute. 

Instead of using dummy variables to model many levels of a categorical variable the authors propose you use an empirical bayes technique to transform them raw categories into continuous values. They provide example for a hierarchical structure as well if you have multilevel data. You need to be careful with overfitting here but it is a very interesting take on the problem.

### R

