#include <set>
#include "PDESolver.h"
#include <igl/jet.h>
#include "igl/is_symmetric.h"
#include "igl/boundary_facets.h"
#include "igl/grad.h"
#include "common.h"

extern Parameter parameter;


void PDESolver::initialize(){
    Prob.setZero(mesh->V.rows(),1);
    Prob(0) = 10;
    // we can show that Laplacian for genus 0 mesh is symmetric, Mass is diagonal
    igl::cotmatrix(mesh->V,mesh->F,Laplacian);
    igl::massmatrix(mesh->V,mesh->F,igl::MASSMATRIX_TYPE_BARYCENTRIC,Mass);
    std::ofstream os;    
    os.open("V.dat");
    os << mesh->V;
    os.close();

    os.open("F.dat");
    os << mesh->F;
    os.close();

}

void PDESolver::solvePoisson(){
//  first step is to set up boundary condition
    Eigen::MatrixXi bF;
    igl::boundary_facets(mesh->F,bF);
    std::set<int> boundary_V;
	for(int i = 0; i < bF.size(); i++){
		boundary_V.insert(bF(i));
	}
    
    boundary_V.insert(985);
    
	Eigen::SparseMatrix<double> A;
	Eigen::Triplet<double> triplets;
    Eigen::VectorXd source;
    source.setZero(mesh->V.rows(),1);

	for (std::set<int>::iterator it=boundary_V.begin(); it!=boundary_V.end(); ++it){
		source(*it) = 0;
	}
    source(985) = 10;

	for (int k = 0; k < Laplacian.outerSize(); k++){
		for (Eigen::SparseMatrix<double>::InnerIterator it(Laplacian,k);it;++it){
			// if contains in the boundary set
			if (boundary_V.find(it.row()) !=boundary_V.end()){
                            Laplacian.coeffRef(it.row(),k) = 0;
			}
		}
                if (boundary_V.find(k) !=boundary_V.end()){
                    Laplacian.coeffRef(k,k) = Mass.coeffRef(k,k);
                }
	}
    
    source = Mass*source;
    std::ofstream os;    
    os.open("laplace.dat");
    os << Laplacian;
    os.close();

    os.open("source.dat");
    os << source;
    os.close();
    
//    int rhs_idx = 0;
//    source(0) = 10;
//    Laplacian.row(0) = 0;
//    Laplacian.coeffRef(0,0) = 1;
    Eigen::VectorXd solution;
//    assert(igl::is_symmetric(Laplacian));
    Eigen::SparseLU<Eigen::SparseMatrix<double > > solver;
    solver.compute(Laplacian);
    assert(solver.info() == Eigen::Success);
    solution = solver.solve(source).eval();
    assert(solver.info() == Eigen::Success);
    
  
    os.open("poisson_solution.dat");
    os << solution;
    os.close();
    
    Eigen::MatrixXd C;
    Eigen::MatrixXd FC;
    FC.setZero(mesh->F.rows(),3);
    igl::jet(solution,true,C);
    Eigen::MatrixXd output;
    for (int i = 0; i < mesh->F.rows();i++){
         FC.row(i) = C.row(mesh->F(i,0))+  C.row(mesh->F(i,1)) + C.row(mesh->F(i,2));
    }
    FC /= 3.0;
    os.open("poisson_color.dat");
    os << FC;
    os.close();
    
    
    Eigen::SparseMatrix<double> G;
    igl::grad(mesh->V,mesh->F,G);
    // G #faces*dim by #V Gradient operator 
    
    // Compute gradient of U
  Eigen::MatrixXd GU = Eigen::Map<const Eigen::MatrixXd>((G*solution).eval().data(),mesh->F.rows(),3);
    
    os.open("poisson_gradient.dat");
    os << GU;
    os.close();
    
}


void PDESolver::solveDiffusion(){
    
    this->solverIntegralType = EulerImplicit;
    nStep = parameter.PDE_nstep;
    dt_ = parameter.PDE_dt;
    std::ofstream os;
    std::cout << "Solver begin" << std::endl;
    for (int s = 0; s < nStep; s++){
        if (solverIntegralType == EulerExplicit){
        // mass matrix is "supposed" to be a diagnoal matrix    
            Prob += dt_ * (Mass.diagonal().inverse()*Laplacian*Prob);
        } else {
            // Solve (mass-dt*L) prob = Mass*prob
        const auto & S = (Mass - dt_*Laplacian);
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
        assert(solver.info() == Eigen::Success);
        Prob = solver.solve(Mass*Prob).eval();
    
        }
        
        if (s == 0 || (s+1)%10 == 0){
              // Compute pseudocolor
            Eigen::MatrixXd C;
            Eigen::MatrixXd FC;
            FC.setZero(mesh->F.rows(),3);
            igl::jet(Prob,true,C);
            Eigen::MatrixXd output;
//            output << Prob;
            std::stringstream ss;
            ss << s;
            os.open("prob_solution"+ss.str()+".dat");
            os << Prob;
            os.close();
            for (int i = 0; i < mesh->F.rows();i++){
                FC.row(i) = C.row(mesh->F(i,0))+  C.row(mesh->F(i,1)) + C.row(mesh->F(i,2));
            }
            FC /= 3.0;
            
            os.open("prob_color"+ss.str()+".dat");
            os << FC;
            os.close();
        }
    }

}
