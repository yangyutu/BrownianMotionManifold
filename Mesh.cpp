#include <cmath>
#include "Mesh.h"
#include <igl/readOFF.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>
#include <igl/adjacency_list.h>
#include "igl/unique_edge_map.h"
#include <iostream>
#include "igl/all_edges.h"
#include "igl/edge_flaps.h"
#include "igl/edges.h"
#include "igl/triangle_triangle_adjacency.h"
#include "Eigen/Geometry"


void Mesh::readMeshFile(std::string filename) {
    // Load a mesh in OFF format
    igl::readOFF(filename, V, F);
    numF = F.rows();
    numV = V.rows();
#ifdef DEBUG
        std::cout << "vertices: " << V << std::endl;
        std::cout << "faces: " << F << std::endl; 
#endif    
    
}

void Mesh::initialize() {
    // calculating normals per face
    igl::per_face_normals(V,F,F_normals);
#ifdef DEBUG
    
        std::cout << "F_normals : " << std::endl;
    for (int i = 0; i < numF; i++){
        std::cout << F_normals.row(i) << std::endl;
    }
#endif      
    
    // self build face edges
    for (int i = 0; i < F.rows(); i++) {
        Eigen::Matrix3d mat;
        for (int j = 0; j < 3; j++) {
            int idx1,idx2;
            idx1 = F(i, j);
            idx2 = F(i, (j + 1) % 3);
            mat.col(j) = (V.row(idx2) - V.row(idx1)).transpose(); 
        }
        this->bases.push_back(V.row(F(i,0)).transpose());
        this->Face_Edges.push_back(mat);
#ifdef DEBUG
        std::cout << "face: " << i << std::endl;
        std::cout << "edges: " << Face_Edges[i] << std::endl; 
#endif
    }

    // build local global local transformations
    for (int i = 0; i < F.rows(); i++) {
        Eigen::Vector3d p10 = (V.row(F(i, 1)) - V.row(F(i, 0))).transpose().eval();
        Eigen::Vector3d p20 = (V.row(F(i, 2)) - V.row(F(i, 0))).transpose().eval();
        Eigen::Matrix<double, 3, 2> J;
        J << p10, p20;
        this->Jacobian_l2g.push_back(J);
        Eigen::Matrix<double, 2, 3> JInv;
        // this is the pinv
        JInv = (J.transpose() * J).inverse().eval() * J.transpose().eval();
#ifdef DEBUG
        double res = (JInv * J - Eigen::MatrixXd::Identity(2,2)).norm();
        if (res > 1e-8){
            std::cout << J << std::endl;
            std::cout << JInv << std::endl;
            std::cerr << "pinv incorrect!: " << res <<std::endl;
        }
#endif
        this->Jacobian_g2l.push_back(JInv);

        Eigen::Vector3d base = V.row(F(i, 0)).transpose().eval();
        this->coord_l2g.push_back(CoordOp_l2g(base, J));
        this->coord_g2l.push_back(CoordOp_g2l(base, JInv));
    }
    
 
    //  construct neighboring faces for each faces in the order    
    igl::triangle_triangle_adjacency(F, TT, TTi);
    //  now calculating the rotation matrix between faces
#if 1
    for (int i = 0; i < numF; i++){
        RotMat.push_back(std::vector<Eigen::Matrix3d>(3,Eigen::MatrixXd::Identity(3,3)));
        this->localtransform_v2v.push_back(std::vector<Eigen::Matrix2d>(3,Eigen::Matrix2d()));
        this->localtransform_p2p.push_back(std::vector<CoordOp_l2l>(3,CoordOp_l2l()));
        
        for (int j = 0 ; j < 3; j++){
            if (TT(i,j) >= 0){
                Eigen::Vector3d normal1 = F_normals.row(i).transpose().eval();
                Eigen::Vector3d normal2 = F_normals.row(TT(i,j)).transpose().eval();               
                Eigen::Vector3d director = Face_Edges[i].col(j).eval();
//                Eigen::Vector3d director = Face_Edges[j].col(TTi(i,j));
                                
                double angle = acos(normal1.dot(normal2));
   
                director /= director.norm();
                
                this->RotMat[i][j] = Eigen::AngleAxisd(angle,director);
#ifdef DEBUG                
                double diff = (RotMat[i][j]*normal1 - normal2).norm();
                if (diff > 1e-6){

                    
                    std::cerr << "rotation matrix incorrect!" << diff << std::endl;
                    
                    std::cout << normal1 << std::endl;
                    std::cout << normal2 << std::endl;
                    std::cout << director << std::endl;  
                    Eigen::Matrix3d mat;
                    mat << normal1, normal2, director;                     
                    std::cout << mat.determinant() << std::endl;
                    std::cout << angle << std::endl;
                    std::cout << RotMat[i][j] << std::endl;

                    
                }
#endif
                this->localtransform_v2v[i][j] = this->Jacobian_g2l[TT(i,j)]*
                        RotMat[i][j]*this->Jacobian_l2g[i];
              
                this->localtransform_p2p[i][j] = CoordOp_l2l(bases[i],bases[TT(i,j)],
                        this->Jacobian_l2g[i],this->Jacobian_g2l[TT(i,j)]);
                
            }
        }
    }
 
    
#endif 
}

bool Mesh::inTriangle(Eigen::Vector2d q){
    if (q(0) >=0 && q(0) <= 1&&q(1) >=0 && q(1) <= 1 && (q(0)+q(1)) <=1 ){
        return true;
    }
//        if (abs(q(0)) >=tol && abs(q(0)-1.0) >= tol && abs(q(1)) <=tol
//            && abs(q(1)-1) <= tol && abs(q(0)+q(1)-1) <=tol ){
//        return true;
//    }
    return false;
}
