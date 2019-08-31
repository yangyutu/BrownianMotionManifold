# BrownianMotionManifold

This is the repo containing our implementation of a general Brownian dynamics simulation algorithm on complex surfaces. We approximate complex surfaces with triangle mesh surfaces and formulate the numerical procedure with respect to triangle mesh surfaces. 
Our algorithm operates in a hybrid manner that we compute forces and velocities in global coordinates but update the positions in local coordinates. 

# Dependence
## Libigl (header only library)



# Usage

## single particle diffusion
./test.exe singleP
change parameters in run.txt

## multiple particle systems with external forces
./test.exe multiP
change parameters in run_multip.txt

## single particle diffusion analysis
./test.exe Diffusion
change parameters in run.txt


# Remarks

In each of folders, we provide example mesh surfaces (.off file) and run.txt/run_multip.txt files.

# References

[[A Simulation Algorithm for Brownian Dynamics on Complex Curved Surfaces]](https://arxiv.org/abs/1908.07166)
