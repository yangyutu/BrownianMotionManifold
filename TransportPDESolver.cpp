#include "TransportPDESolver.h"
#include <igl/jet.h>







void TransportPDESolver::initialize(){
    Prob.setZero(mesh->V.rows(),1);
    Prob(0) = 10;
    igl::cotmatrix(mesh->V,mesh->F,Laplacian);
    igl::massmatrix(mesh->V,mesh->F,igl::MASSMATRIX_TYPE_BARYCENTRIC,Mass);
    

}


void TransportPDESolver::solve(){
    
    this->solverIntegralType = TransportPDESolver::EulerImplicit;
    nStep = 100;
    dt_ = 0.1;
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