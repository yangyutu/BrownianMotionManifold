#pragma once

#include "model.h"
#include "common.h"


class Model_cell:public Model{
public:
	Model_cell();
	~Model_cell(){}

	struct CellInfo{
		Eigen::Vector3d polarization, inhibition;
                std::vector<int> nbList;
	};

	virtual void moveOnMesh(int p_idx);
	virtual void run();
	typedef std::shared_ptr<CellInfo> CellInfo_ptr;
    std::vector<CellInfo_ptr> cellInfo;

protected:
	virtual void calForces();
    virtual void calForcesHelper(int i, int j, Eigen::Vector3d &F,Eigen::Vector3d &inhit);
    void buildNbList();
    virtual void outputTrajectory(std::ostream &os);

private:
	double tau,sigma,V_a,V_r,D0,beta,polar_vel;
};
