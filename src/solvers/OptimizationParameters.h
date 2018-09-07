//============================================================================
// Name        : OptimizationParameters.h
// Author      : Yaser
// Version     :
// Copyright   :
// Description : Solver parameters
//============================================================================


#ifndef PARAMETER_H
#define PARAMETER_H

class OptimizationParameters {
public:
	long int m;
	long int n;
	long int num_iterations;  // maximum number of iterations as the stopping criteria
	int lbfgs_memory;   // memory of LBFGS
	double stepsize_init;  // stepsize for the first step (gradient descent step)
	double stepsize;   // stepsize in Quasi-Newton algorithm
	int verbose;
};

#endif /* PARAMETER_H */
