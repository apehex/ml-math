# Losses

Computing the function norms introduced in [the approximation theorems](../theorems/README.md) is challenging:
- they often require to evaluate the function on an *infinite* input space
- the data is simply not available

Instead, heuristics called "loss functions" are computed on a finite set of data points.

They quantify the quality of an approximation with tools from the probability theory, algebra, etc.

And they have useful properties to allow the search for a good approximation: namely, they are differentiable.
