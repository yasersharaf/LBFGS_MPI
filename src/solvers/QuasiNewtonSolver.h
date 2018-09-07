//============================================================================
// Name        : QuasiNewtonSolver.h
// Author      : Yaser
// Version     :
// Copyright   :
// Description : Class of quasi newton solvers
//============================================================================


#ifndef QUASINEWTONSOLVER_H_
#define QUASINEWTONSOLVER_H_

using namespace std;

#include "ProblemData.h"
#include "OptimizationParameters.h"

class QuasiNewtonSolver {
protected:
	OptimizationParameters& optParams;



public:

	QuasiNewtonSolver(ProblemData _problemData, OptimizationParameters _optParams) :
			optParams(_optParams)  {

	}

	virtual void computeAndTakeStep();
	virtual double computeCurrentObjectiveValue();
	virtual ~QuasiNewtonSolver(){};

};

#endif /* QUASINEWTONSOLVER_H_ */
