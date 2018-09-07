//============================================================================
// Name        : LBFGSSolver.cpp
// Author      : Yaser
// Version     :
// Copyright   :
// Description : LBFGS MPI implementation
//============================================================================

#include "LBFGSSolver.h"
#include <boost/mpi.hpp>
#include <iomanip>


namespace mpi = boost::mpi;

//virtual void computeAndTakeStep();
//virtual double ();

LBFGSSolver::LBFGSSolver(ProblemData problemData, OptimizationParameters _optParams) :
						QuasiNewtonSolver(problemData, _optParams) {
	mpi::communicator world;
	stepSize = optParams.stepsize;
	lbfgsHistorySize = optParams.lbfgs_memory;
	m = optParams.m;
	globalN = optParams.n;
	localN = optParams.n / world.size();
	int localNTmp = localN;
	// We are going to split data by columns. Local N tells how many columns
	//	are stored locally.
	if (world.rank() == world.size() - 1) { // If optParam.n is not divisible by world.size we need to have few extra points on last node
		localN = optParams.n - (localN * (world.size() - 1));
	}
	Alocal.resize(localN * m);
	xLocalEven.resize(localN);
	xLocalOdd.resize(localN);
	cLocal.resize(localN * lbfgsHistorySize*2);
	preMatrixGlobal.resize((2*lbfgsHistorySize+1) * (2*lbfgsHistorySize+1));
	gradientLocalEven.resize(localN);
	gradientLocalOdd.resize(localN);
	xGlobalEven.resize(globalN);
	xGlobalOdd.resize(globalN);
	pK.resize(globalN);
	bLocal.resize(m);
	// Send data to other nodes
	if (world.rank() == 0) { // Sends data to others
/*
		for (int r = 0; r < m; r++) {
			for (int c = 0; c < globalN; c++) {
				cout << problemData.A[r + c * m] << " ";
			}
			cout << endl;
		}
*/
		for (int i = 1; i < world.size(); i++) {
			int nnz = localNTmp;
			if (i == world.size() - 1) {
				nnz = optParams.n - (localNTmp * (world.size() - 1));
			}
			world.send(i, 0, &problemData.A[(i) * localNTmp * m], nnz * m);

		}
		// Also copy local data
		for (int i = 0; i < localN * optParams.m; i++) {
			Alocal[i] = problemData.A[i];
		}
		for (int i = 0; i < m; i++) {
			bLocal[i] = problemData.b[i];
			cout << bLocal[i] << " ";
		}
		cout << endl;
		if (iter%2==0){
			for (int i = 0; i < globalN; i++) {
				xGlobalEven[i] = problemData.x[i];
			}
		} else {
			for (int i = 0; i < globalN; i++) {
				xGlobalOdd[i] = problemData.x[i];
			}
		}

	} else {
		// Receive data from root
		world.recv(0, 0, &Alocal[0], localN * m);

	}
	if (iter%2==0){
		this->sendXFromRootToNodes(xGlobalEven);
	} else {
		this->sendXFromRootToNodes(xGlobalOdd);
	}

}

void LBFGSSolver::sendXFromRootToNodes(std::vector<double> xIn) {

	mpi::communicator world;
	if (world.rank() == 0) { // Sends data to others
		for (int i = 1; i < world.size(); i++) {
			int nnz = localN;
			if (i == world.size() - 1) {
				nnz = globalN - (localN * (world.size() - 1));
			}
			world.send(i, 1, &xIn[(i) * localN], nnz);

		}
		// Also copy local data
		if (iter%2==0){

			for (int i = 0; i < localN; i++) {
				xLocalEven[i] = xIn[i];
			}
		} else {
			for (int i = 0; i < localN; i++) {
				xLocalOdd[i] = xIn[i];
			}
		}

	} else {
		// Receive data from root
		if (iter%2==0){
			world.recv(0, 1, &xLocalEven[0], localN);
		} else {
			world.recv(0, 1, &xLocalOdd[0], localN);

		}
	}

}


std::vector<double> LBFGSSolver::computeResidualForCurrentX() { // computes Ax-b
	mpi::communicator world;
	std::vector<double> g(m, 0);

	if (iter%2==0){

		for (int i = 0; i < localN; i++) {
			for (int j = 0; j < m; j++) {
				g[j] += xLocalEven[i] * Alocal[i * m + j];
			}
		}
	} else {

		for (int i = 0; i < localN; i++) {
			for (int j = 0; j < m; j++) {
				g[j] += xLocalOdd[i] * Alocal[i * m + j];
			}
		}
	}

	std::vector<double> gTmp(m, 0);
	mpi::reduce(world, &g[0], m, &gTmp[0], std::plus<double>(), 0);

	if (world.rank() == 0) {
		for (int i = 0; i < m; i++) {
			gTmp[i] -= bLocal[i];
		}
	}
	return gTmp;
}

double LBFGSSolver::computeCurrentObjectiveValue() {
	double objVal = 0;
	// objVal = 1/2 * \|Ax - b\|^2
	// first I multiply  g = ALocal*xLocal
	std::vector<double> residual = this->computeResidualForCurrentX();
	mpi::communicator world;

	if (world.rank() == 0) {

		for (unsigned int i = 0; i < residual.size(); i++) {
			objVal += residual[i] * residual[i];
		}
	}
	return objVal / 2;
}
std::vector<double> LBFGSSolver::computeGradientForCurrentX() { // computes A'*(Ax-b)

	std::vector<double> residual = this->computeResidualForCurrentX();
	mpi::communicator world;
	if (world.rank() == 0) {
		//cout<<"residual: "<< residual[0]<< "  iter: "<<iter<<" xLocalEven: "<<xLocalEven[0]<<" xLocalOdd: "<<xLocalOdd[0]<<endl;
	}



	broadcast(world, &residual[0], residual.size(), 0);  // now residuals is the same on each node.

	// gradient = A'*residual
	if(iter%2 == 0){
		for (int r = 0; r < m; r++) {
			for (int col = 0; col < localN; col++) {
				gradientLocalEven[col] = 0;
			}
		}

		for (int r = 0; r < m; r++) {
			for (int col = 0; col < localN; col++) {
				gradientLocalEven[col] += residual[r] * Alocal[r + col * m];
			}
		}
	} else {
		for (int r = 0; r < m; r++) {
			for (int col = 0; col < localN; col++) {
				gradientLocalOdd[col] = 0;
			}
		}
		for (int r = 0; r < m; r++) {
			for (int col = 0; col < localN; col++) {
				gradientLocalOdd[col] += residual[r] * Alocal[r + col * m];
			}
		}
	}

	// gather to a root note to store full gradient
	std::vector<double> gradient;
	if (world.rank() == 0) {
		gradient.resize(globalN);
		for (int from = 1; from < world.size(); from++) {
			int nnz = localN;
			if (from == world.size() - 1) {
				nnz = globalN - (localN * (world.size() - 1));
			}
			world.recv(from, 0, &gradient[localN * from], nnz);

		}
		if(iter%2 == 0){

			for (int i = 0; i < localN; i++) {
				gradient[i] = gradientLocalEven[i];
			}
		} else {
			for (int i = 0; i < localN; i++) {
				gradient[i] = gradientLocalOdd[i];
			}
		}

	} else {
		// send data to ROOT
		if(iter%2 == 0){
			world.send(0, 0, &gradientLocalEven[0], gradientLocalEven.size());
		} else {
			world.send(0, 0, &gradientLocalOdd[0], gradientLocalOdd.size());
		}
	}

	return gradient;

}


std::vector<double> LBFGSSolver::computeMatrixPre() { // calculates the b.b' (here c.c') and returns precondition matrix of alg. 3
	mpi::communicator world;
	// First time to enter this function, iter=0
	if (iter>0){
		int mod1 = (iter-1) % lbfgsHistorySize;

		int mod2 = (iter-1) % lbfgsHistorySize + lbfgsHistorySize;

		std::vector<double> preMatUpdateLocal(2*lbfgsHistorySize+1, 0);



		for (int i = 0; i < localN; i++) {
			for (int j = 0; j < 2*lbfgsHistorySize; j++) {
				preMatUpdateLocal[j] += cLocal[localN*mod1+ i] * cLocal[j * localN + i];
			}
		}


		//std::vector<double> gTmp(2*lbfgsHistorySize, 0); ////// why?
		mpi::reduce(world, &preMatUpdateLocal[0], 2*lbfgsHistorySize,
				&preMatrixGlobal[mod1*(2*lbfgsHistorySize+1)], std::plus<double>(), 0);

		// mod1*(2*lbfgsHistorySize) is the first element in the column 3
		std::fill(preMatUpdateLocal.begin(), preMatUpdateLocal.end(), 0);     //
		if (world.rank() == 0) {    //
			for (int i = 0; i < 2*lbfgsHistorySize; i++) {
				preMatrixGlobal[mod1 + i * (2*lbfgsHistorySize+1) ]	=
						preMatrixGlobal[mod1*(2*lbfgsHistorySize+1)+i];
			}
		}


		for (int i = 0; i < localN; i++) {
			for (int j = 0; j < 2*lbfgsHistorySize; j++) {
				preMatUpdateLocal[j] += cLocal[localN*mod2+ i] * cLocal[j * localN + i];
			}
		}


		mpi::reduce(world, &preMatUpdateLocal[0], 2*lbfgsHistorySize, &preMatrixGlobal[mod2*(2*lbfgsHistorySize+1)], std::plus<double>(), 0);
		std::fill(preMatUpdateLocal.begin(), preMatUpdateLocal.end(), 0);     //
		// mod2(?)*(2*lbfgsHistorySize) is the first element in the column 3

		if (world.rank() == 0) {    ////////////////////////////////
			for (int i = 0; i < 2*lbfgsHistorySize; i++) {
				preMatrixGlobal[mod2 + i * (2*lbfgsHistorySize +1)]	= 	preMatrixGlobal[mod2*(2*lbfgsHistorySize+1)+i];
			}
		}



		if (iter%2 == 0){
//			cout<<"grad @ even= [";
			for (int i = 0; i < localN; i++) {
				for (int j = 0; j < 2*lbfgsHistorySize; j++) {
					preMatUpdateLocal[j] += gradientLocalEven[i] * cLocal[j * localN + i];
				}
				preMatUpdateLocal[2*lbfgsHistorySize] += gradientLocalEven[i]*gradientLocalEven[i];
//				cout<<gradientLocalOdd[i]<<"   "<<gradientLocalEven[i];

			}
//			cout<<endl;
//			cout<<"gradNorm @ even= "<<preMatUpdateLocal[2*lbfgsHistorySize]<<endl;

		}
		else {
//			cout<<"grad @ odd= [";
			for (int i = 0; i < localN; i++) {
				for (int j = 0; j < 2*lbfgsHistorySize; j++) {
					preMatUpdateLocal[j] += gradientLocalOdd[i] * cLocal[j * localN + i];
				}
				preMatUpdateLocal[2*lbfgsHistorySize] += gradientLocalOdd[i]*gradientLocalOdd[i];

			}
		}

		//std::vector<double> gTmp(2*lbfgsHistorySize, 0);  ////// why?
		mpi::reduce(world, &preMatUpdateLocal[0], 2*lbfgsHistorySize+1,
				&preMatrixGlobal[(2*lbfgsHistorySize+1)*(2*lbfgsHistorySize)], std::plus<double>(), 0);
		// mod1*(2*lbfgsHistorySize) is the first element in the column 3

		if (world.rank() == 0) {    ////////////////////////////////
			for (int i = 0; i < 2*lbfgsHistorySize; i++) {
				preMatrixGlobal[2*lbfgsHistorySize + i * (2*lbfgsHistorySize+1) ]	=
						preMatrixGlobal[(2*lbfgsHistorySize+1)*(2*lbfgsHistorySize)+i];
			}
		}
	}
	return preMatrixGlobal;

}


void LBFGSSolver::findPk() { // finds p_k
	std::vector<double> preMatrix = this->computeMatrixPre();
	mpi::communicator world;

	std::vector<double> delta(2*lbfgsHistorySize+1, 0);

	if (world.rank() == 0) {


		std::vector<double> alpha(lbfgsHistorySize, 0);
		delta[2*lbfgsHistorySize] = -1;
		int last_i = max(iter - lbfgsHistorySize, 0);
		// first loop

//		cout<<"first loop: prematrix access"<<endl;
		for (int i = iter - 1; i >= last_i; i--) {
//			int j = i - (iter-lbfgsHistorySize) + 1;
			int j = i%lbfgsHistorySize;

			for (int l = 0; l<2*lbfgsHistorySize+1;l++){
				alpha[j] += preMatrix[(2*lbfgsHistorySize+1)*j+l]*delta[l];
//				cout<<"[j,l] = ["<<j<<","<<l<<"]   ";

			}
			alpha[j] = alpha[j]/preMatrix[j*(2*lbfgsHistorySize+1)+(lbfgsHistorySize+j)];
//			cout<<"\ns[j,m+j] = ["<<j<<","<<lbfgsHistorySize+j<<"]   ";
			delta[lbfgsHistorySize+j] -=  alpha[j];

//			cout<<endl<<"next mem";

		}


		double scalarMultiplier = preMatrix[((iter - 1)%lbfgsHistorySize)*(2*lbfgsHistorySize+1)+
		                                    (iter - 1)%lbfgsHistorySize+lbfgsHistorySize]/preMatrix[((iter - 1)%lbfgsHistorySize+lbfgsHistorySize)*(2*lbfgsHistorySize+1)+((iter - 1)%lbfgsHistorySize+lbfgsHistorySize)];

		for (int i = 0; i<2*lbfgsHistorySize+1; i++) {
			delta[i] *= scalarMultiplier;
		}

		// second loop
		double bbeta = 0;
//		cout<<"second loop: prematrix access"<<endl;
		int first_i = max(0, iter - lbfgsHistorySize);
		for (int i = first_i; i < iter; i++) {
			bbeta = 0;
//			int j = i - (iter-lbfgsHistorySize) + 1;
			int j = i%lbfgsHistorySize;

			for (int l = 0; l<2*lbfgsHistorySize+1;l++){
				bbeta += preMatrix[(2*lbfgsHistorySize+1)*(lbfgsHistorySize+j)+l]*delta[l];
			}
			bbeta =  bbeta /preMatrix[j*(2*lbfgsHistorySize+1)+(lbfgsHistorySize+j)];
			delta[j] +=  alpha[j]-bbeta;

		}


	}
	broadcast(world, &delta[0], delta.size(), 0);  // now delta is the same on all nodes.
	std::vector<double> pLocal (localN);
	for (int i = 0; i < localN ; i++){
		for (int j = 0; j< 2*lbfgsHistorySize ; j++){
			pLocal[i] += delta[j] * cLocal[j*localN+i];
		}
	}






	if (iter%2 == 0){
		for (int i = 0; i < localN ; i++){
			pLocal[i] += delta[2*lbfgsHistorySize] * gradientLocalEven[i];
		}
	} else {
		for (int i = 0; i < localN ; i++){
			pLocal[i] += delta[2*lbfgsHistorySize] * gradientLocalOdd[i];
		}
	}

//	std::vector<double> pK ;

	if (world.rank() == 0) {
//		pK.resize(globalN);
		for (int from = 1; from < world.size(); from++) {
			int nnz = localN;
			if (from == world.size() - 1) {
				nnz = globalN - (localN * (world.size() - 1));
			}
			world.recv(from, 0, &pK[localN * from], nnz);

		}

		for (int i = 0; i < localN; i++) {
			pK[i] = pLocal[i];
		}


	} else {
		// send data to ROOT
		world.send(0, 0, &pLocal[0], localN);
		if (world.rank()==1){
//			cout<<endl<<endl<<endl<<"pLocal on node "<<world.rank()<<" is:   "<<pLocal[0]<<endl;
		}


	}


}


void LBFGSSolver::updateYk() { // finds y_k after each iteration
	mpi::communicator world;
	int modd = iter % lbfgsHistorySize;
	if (iter%2 == 0){
		for(int i =0 ; i< localN; i++){
			cLocal[(lbfgsHistorySize+modd)*localN + i] = gradientLocalOdd[i] - gradientLocalEven[i];
		}
	} else {
		for(int i =0 ; i< localN; i++){
			cLocal[(lbfgsHistorySize+modd)*localN + i] = gradientLocalEven[i] - gradientLocalOdd[i];
		}
	}
}

void LBFGSSolver::updateSk() { // finds s_k after each iteration
	mpi::communicator world;
	// First time to enter this function, iter=0
	int modd = iter % lbfgsHistorySize;
	if (iter%2 == 0){
		for(int i =0 ; i< localN; i++){
			cLocal[modd*localN + i] = xLocalOdd[i] - xLocalEven[i];	// bb local is stack of y_k, s_k and grad_k
		}
	} else {
		for(int i =0 ; i< localN; i++){
			cLocal[modd*localN + i] = xLocalEven[i] - xLocalOdd[i];
		}
	}
}


void LBFGSSolver::computeAndTakeStep() {
	mpi::communicator world;

	if(iter == 0){
		gradientEven = this->computeGradientForCurrentX();
	}
	/*        two loop recursion	 */
	if (iter>=0){
		this->findPk();
	}

	/*      Take the step and update s_k and y_k   */


	if(iter%2 == 0){

		if (world.rank() == 0) {
			for (int i = 0; i < globalN; i++) {
				if (iter==0){
					xGlobalOdd[i] = xGlobalEven[i] - stepSize * gradientEven[i];
				}else{
					xGlobalOdd[i] = xGlobalEven[i] + stepSize * pK[i];
				}
			}
			//cout<<endl<<"gradientEven: "<< gradientEven[0]<< "  iter: "<<iter<<" xGlobal even cur: "<<xGlobalEven[0]<<" xGlobal odd next: "<<xGlobalOdd[0]<<endl;
		}

		iter +=1;
		this->sendXFromRootToNodes(xGlobalOdd);



		iter -=1;
		this->updateSk();
		iter +=1;

		gradientOdd = this->computeGradientForCurrentX();

		iter -=1;
		this->updateYk();
		iter +=1;


	} else {
		if (world.rank() == 0) {

			for (int i = 0; i < globalN; i++) {
				if (iter==0){
					xGlobalEven[i] = xGlobalOdd[i]- stepSize * gradientOdd[i];
				}else{
					xGlobalEven[i] = xGlobalOdd[i]+ stepSize * pK[i];
				}
			}

		}

		iter +=1;
		this->sendXFromRootToNodes(xGlobalEven);

		iter -=1;
		this->updateSk();
		iter +=1;
		gradientEven = this->computeGradientForCurrentX();
		iter -=1;
		this->updateYk();
		iter +=1;

	}


}
