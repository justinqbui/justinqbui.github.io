---
title: "CSE 142 - Machine Learning Notes"
excerpt_separator: "<!--more-->"
classes: "wide"
categories:
  - Notes

tags:
  - 
  - 
---

## Key ML concepts
An ML system is a **task** requires an appropriate mapping - a **model** - from data described by **features** to outputs. Obtaining such a model from training data is called a **learning problem**.

**Dimensionality:** ML problems are generally dealt with in high-dimensional data. We use **distance measures** to measure similarity between embedding vectors.
  - $L1$ distance (Manhattan): $d(x,y) = \sum_{i=1}^d \mid x_i - y_i \mid$
  - $L2$ distance (Euclidian):  $d(x,y) =\; \mid\mid x - y \mid\mid \;= (\sum_{i=1}^d (x_i - y_i)^2)^{1/2}$
  - $L_p$ distance (Minkowski): $d(x,y) = (\sum_{i=1}^d (x_i - y_i)^p)^{1/p}$

<p align="center">
  <img src="/images/cse142/Lp distance.png" width = "80%">
</p>

**Bayes Rule:** $P(X|Y) = \frac{P(Y|X)(P(X)}{P(Y)}$  
  - Usually framed in terms of hypotheses $H$ and data $D$
    - $P(H_i\|D) = \frac{P(D\|H_i)(P(H_i)}{P(D)}$
    - Posterior probability (causal knowledge) = $P(H_i\|D)$
    - Likelihood (diagnostic knowledge) = $P(D\|H_i)(P(H_i)$
    - Prior probability = $P(H_i)$
    - Normalizing Constant = $P(D)$

**Geometric model:** use intuitions from geometry such as separating (hyper-)planes, linear transformations, and distance metrics  
**Probabilistic models:** view learning as a process of reducing uncertainty, modelled by means of probability distributions  
**Logical models:** are defined in terms of easily interpretable logical expressions  

**Grouping models:** divide the *instance space* (the space of possible inputs) into segments, in eah segment a different (perhaps very simple) model is learned  
**Grading models:** learning a single, global model over the instace space

## Linear Classification