//============================================================================
// Name        : ProblemData.h
// Author      : Yaser
// Version     :
// Copyright   :
// Description : Optimization problem parameters
//============================================================================


#ifndef PROBLEMDATA_H_
#define PROBLEMDATA_H_

#include <vector>
class ProblemData {

public:
// parametes for a least square problem min ||Ax-b||
	std::vector<double> b;
	std::vector<double> A; // Matrix A is in column major
	std::vector<double> x;
};

#endif /* PROBLEMDATA_H_ */
