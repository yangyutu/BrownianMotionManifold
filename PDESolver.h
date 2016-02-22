#pragma once

#include "Mesh.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readDMAT.h>
#include <igl/readOFF.h>
#include <igl/repdiag.h>
//#include <igl/viewer/Viewer.h>
#include <memory>
#include <iostream>
#include <fstream>

class PDESolver{
public:
    enum SolverIntegralType{EulerExplicit=0,EulerImplicit=1};
    PDESolver(){}
    void solveDiffusion();
    void solvePoisson();
    void initialize();
    void readMeshFile();
    std::shared_ptr<Mesh> mesh;
   
private:
    Eigen::SparseMatrix<double> Laplacian;
    Eigen::SparseMatrix<double> Mass;
    Eigen::VectorXd Prob;
    double dt_;
    int nStep;
    int solverIntegralType;

};
