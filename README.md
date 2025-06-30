# powersmooth
Python implementation of powersmooth (https://de.mathworks.com/matlabcentral/fileexchange/48799-powersmooth), invented by B. M. Friedrich.

The library provides tools for smoothing one dimensional signals using
derivative based regularisation.  Two helper functions are included to create
grids suitable for the smoothing routine:

* ``upsample_with_mask`` – inserts additional points between existing samples
  while keeping track of the original locations.
* ``upsample_to_uniform`` – resamples data on a uniformly spaced grid via
  linear interpolation and returns a mask marking the original sample
  positions.
