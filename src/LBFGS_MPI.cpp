//============================================================================
// Name        : LBFGS_MPI.cpp
// Author      : Yaser
// Version     :
// Copyright   : 
// Description : LBFGS MPI implementation
//============================================================================

/*
 ** consider quadratic function, f(x) = 0.5*\|Ax-b\|^2
 ** Implement LBFGS algorithm as is explained in
 ** https://en.wikipedia.org/wiki/Limited-memory_BFGS
 **
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>  /* rand, srand */
#include <vector>

#include <time.h>
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;

#include <boost/timer/timer.hpp>

#include "solvers/OptimizationParameters.h"
#include "solvers/ProblemData.h"
#include "solvers/QuasiNewtonSolver.h"
#include "solvers/LBFGSSolver.h"

using namespace std;

int main(int argc, char* argv[]) {
	mpi::environment env(argc, argv);
	mpi::communicator world;
	srand(1);
	ProblemData problemData;
	OptimizationParameters optParams;
	optParams.m = 1000;
	optParams.n = 2000;
	optParams.num_iterations = 10000;//100 initially
	optParams.lbfgs_memory = 100; //10 initially
	optParams.stepsize = 10 / (0.0 + optParams.n);
	if (world.rank() == 0) { //Generate data on node 0
		int nnz = optParams.m * optParams.n;
		problemData.A.resize(nnz);
		for (int i = 0; i < optParams.n; i++) {
			double tmp = 0;
			for (int j = 0; j < optParams.m; j++) {
				double r = rand() / (RAND_MAX + 0.0);
				problemData.A[i * optParams.m + j] = r;
				tmp += r * r;
			}
			tmp = 1 / sqrt(tmp);
			for (int j = 0; j < optParams.m; j++) {
				problemData.A[i * optParams.m + j] = problemData.A[i * optParams.m + j] * tmp;
			}

		}
		problemData.b.resize(optParams.m);
		for (int i = 0; i < optParams.m; i++) {
			problemData.b[i] = rand() / (RAND_MAX + 0.0);
		}
		problemData.x.resize(optParams.n, 0);
	}

/*
	if (world.rank() == 0) {
		cout<<"Amatrix: "<<endl;
		for (int i = 0; i < optParams.n; i++) {
			for (int j = 0; j < optParams.m; j++) {
				cout<< std::setprecision(17)<<problemData.A[i*optParams.m+j]<<"     ";
			}
			cout<<endl;
		}

		cout<<"endA"<<endl;
		cout<<"bvec"<<endl;

		for (int i = 0; i < optParams.m; i++) {
			cout<< std::setprecision(17)<<problemData.b[i]<<"     ";
		}
		cout<<"bvecEnd"<<endl;
	}
*/

	boost::timer::cpu_timer timer;
	LBFGSSolver solver(problemData, optParams);
	boost::timer::cpu_times elapsedInInitialization = timer.elapsed();
	if (world.rank() == 0) { // Writes total running time
		std::cout << "Initialization of class took: " << (elapsedInInitialization.user + elapsedInInitialization.system) / 1e9 << " (sec)" << "  Wallclock time: "
				<< elapsedInInitialization.wall / 1e9 << " (sec)" << std::endl;
	}

	/* this function computes and takes the step to obtain a new iterate */
	double objVal = solver.computeCurrentObjectiveValue();
	if (world.rank() == 0) {
		cout << "Initial Obj Value is " << objVal << endl;
	}

	for (int it = 0; it < optParams.num_iterations; it++) {

		timer.start();
		solver.computeAndTakeStep(); // this function computes and takes the step to obtain a new iterate
		boost::timer::cpu_times computingStep = timer.elapsed();

		timer.start();
		objVal = solver.computeCurrentObjectiveValue(); // this function computes and takes the step to obtain a new iterate
		boost::timer::cpu_times objValTime = timer.elapsed();

		if (world.rank() == 0) { // Writes total running time
			std::cout << "Iteration: " << it << " ObjVal: " << objVal << " Times: " << (computingStep.user + computingStep.system) / 1e9 << " (sec)"
					<< "  Wall clock time: " << computingStep.wall / 1e9 << " (sec)" << " " << (objValTime.user + objValTime.system) / 1e9 << " (sec)"
					<< "  Wall clock time: " << objValTime.wall / 1e9 << " (sec)" << std::endl;
		}

	}

	return 0;
}
