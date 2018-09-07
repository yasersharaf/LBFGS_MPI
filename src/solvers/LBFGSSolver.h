//============================================================================
// Name        : LBFGSSolver.h
// Author      : Yaser
// Version     :
// Copyright   :
// Description : LBFGS MPI implementation
//============================================================================


#ifndef HIG213_LBFGSSOLVER_H_
#define HIG213_LBFGSSOLVER_H_

#include "QuasiNewtonSolver.h"
#include <deque>

#include <vector>

class LBFGSSolver: public QuasiNewtonSolver {

private:
	std::deque<std::vector<double> > historyY;
	std::deque<std::vector<double> > historyS;

	int localN;
	int globalN;
	int lbfgsHistorySize;
	double stepSize;
	int iter=0;

	std::vector<double> Alocal;
	
	std::vector<double> xGlobalEven;
	std::vector<double> xGlobalOdd;

	std::vector<double> xLocalEven;
	std::vector<double> xLocalOdd;

	std::vector<double> cLocal; // s_k,y_k

	std::vector<double> preMatrixGlobal; // Global c * cT

	std::vector<double> bLocal;
	
	std::vector<double> sLocal;
	std::vector<double> yLocal;
	std::vector<double> gradientEven;
	std::vector<double> gradientOdd;
	
	std::vector<double> gradientLocalEven;
	std::vector<double> gradientLocalOdd;
	std::vector<double> pK;

	



	std::vector<double> computeResidualForCurrentX();
	std::vector<double> computeGradientForCurrentX();
	std::vector<double> computeMatrixPre();
	void findPk();

	void sendXFromRootToNodes(std::vector<double> x);
	void updateSk();
	void updateYk();


	int m;

public:
	LBFGSSolver(ProblemData _problemData, OptimizationParameters _optParams);
	double computeCurrentObjectiveValue();
	void computeAndTakeStep();
	~LBFGSSolver(){};

};

#endif /* HIG213_LBFGSSOLVER_H_ */
