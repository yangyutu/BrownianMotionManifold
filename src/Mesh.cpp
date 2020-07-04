#include <cmath>
#include "Mesh.h"
#include <igl/readOFF.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>
#include <igl/adjacency_list.h>
#include "igl/unique_edge_map.h"
#include "igl/doublearea.h"
#include <iostream>
#include "igl/all_edges.h"
#include "igl/edge_flaps.h"
#include "igl/edges.h"
#include "igl/triangle_triangle_adjacency.h"
#include "Eigen/Geometry"
#include <cmath>

void Mesh::readMeshFile(std::string filename) {
    // Load a mesh in OFF format
    std::cout << "read mesh file:" << filename << std::endl; 
    igl::readOFF(filename, V, F);
    numF = F.rows();
    numV = V.rows();
    igl::doublearea(V,F,dblA);
    std::cout << "average double area " << dblA.mean() << std::endl;
    std::cout << "total area " << dblA.sum() << std::endl;
    area_total = dblA.sum();
    area_avg = dblA.mean();
    
#ifdef DEBUG2
        std::cout << "vertices: " << V << std::endl;
        std::cout << "faces: " << F << std::endl;
        
#endif    
    
}

void Mesh::initialize() {
    
    
    // calculating normals per face
    igl::per_face_normals(V,F,F_normals);
#ifdef DEBUG2
    
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
#ifdef DEBUG2
        std::cout << "face: " << i << std::endl;
        std::cout << "edges: " << Face_Edges[i] << std::endl; 
#endif
    }

    // build local global local transformations
    for (int i = 0; i < F.rows(); i++) {
        Eigen::Vector3d p10 = (V.row(F(i, 1)) - V.row(F(i, 0))).transpose().eval(); // p1 - p0 (3d)
        Eigen::Vector3d p20 = (V.row(F(i, 2)) - V.row(F(i, 0))).transpose().eval(); // p2 - p0 (3d)
        Eigen::Matrix<double, 3, 2> J;
        J << p10, p20;
        this->Jacobian_l2g.push_back(J);
        Eigen::Matrix<double, 2, 3> JInv;
        // this is the pinv
        JInv = (J.transpose() * J).inverse().eval() * J.transpose().eval(); // J^{-1} = (J^TJ)^{-1} J^T
#ifdef DEBUG2
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
    
 
  //  construct neighboring faces for each faces in the order    (https://github.com/libigl/libigl/blob/master/include/igl/triangle_triangle_adjacency.h)
  // Constructs the triangle-triangle adjacency matrix for a given
  // mesh (V,F).
  //
  // Inputs:
  //   F  #F by simplex_size list of mesh faces (must be triangles)
  // Outputs:
  //   TT   #F by #3 adjacent matrix, the element i,j is the id of the triangle
  //        adjacent to the j edge of triangle i
  //   TTi  #F by #3 adjacent matrix, the element i,j is the id of edge of the
  //        triangle TT(i,j) that is adjacent with triangle i
  //
  // NOTE: the first edge of a triangle is [0,1] the second [1,2] and the third
  //       [2,3].  this convention is DIFFERENT from cotmatrix_entries.h
    igl::triangle_triangle_adjacency(F, TT, TTi);

    //  now calculating the rotation matrix between faces, because local to local transformation requires rotation matrix
    //  the transformation of local velocity to another local velocity has the following procedures
    //  first convert a local tangent velocity to its global one (i.e. the lab frame)
    //  then project this global velocity to the tangent plane of the new surface ()
    //  convert the now tangent velocity to its local version

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
                                
                double proj = normal1.dot(normal2);
                double angle;
                if (proj > 1){
                    angle = 0.0;
                } else if(proj < -1){
                    angle = -M_PI;
                } else {
                    angle = acos(proj);
                }
                director /= director.norm();
                
                this->RotMat[i][j] = Eigen::AngleAxisd(angle,director);
               
                double diff = (RotMat[i][j]*normal1 - normal2).norm();
                double diff3 = (RotMat[i][j]*normal2 - normal1).norm();

                // here is sanity check, after the transformation, diff should be zero.
                if (diff > 1e-6){

                    
//                    std::cerr << "rotation matrix incorrect!" << diff << std::endl;
                    
//                    std::cout << normal1 << std::endl;
//                    std::cout << normal2 << std::endl;
//                    std::cout << director << std::endl;  
                    Eigen::Matrix3d mat;
                    mat << normal1, normal2, director;  
//                    std::cout << "determinant should be positive" << std::endl;
//                    std::cout << mat.determinant() << std::endl;
//                    std::cout << angle << std::endl;
//                    std::cout << RotMat[i][j] << std::endl;
                    
                    // here we try to transpose to fix it because we rotate in the opposite direction!
                    RotMat[i][j].transposeInPlace();
                    double diff2 = (RotMat[i][j]*normal1 - normal2).norm();
                    
                    if (diff2 > 1e-6) {
                        std::cerr << "transpose R to fix it! but failed:" << diff2 << std::endl;
                    } else{
                        //std::cout << "transpose R to fix it" << std::endl;
                    }
                    //double diff = (RotMat[i][j]*normal1 - normal2).norm();
//                    
//                    std::cout << diff << std::endl;
                }
                this->localtransform_v2v[i][j] = this->Jacobian_g2l[TT(i,j)]*
                        RotMat[i][j]*this->Jacobian_l2g[i];
              
                for (int iii=0;iii < 2; iii++){
                    for (int jjj=0; jjj < 2; jjj++){
                        if(std::isnan((localtransform_v2v[i][j])(iii,jjj))){
                            std::cerr <<localtransform_v2v[i][j] << std::endl;
                        std::cerr <<RotMat[i][j] << std::endl;
                        std::cerr <<Jacobian_g2l[TT(i,j)] << std::endl;
                        std::cerr <<Jacobian_l2g[i] << std::endl;
                        }
                    }
                
                }
                
                
                this->localtransform_p2p[i][j] = CoordOp_l2l(bases[i],bases[TT(i,j)],
                        this->Jacobian_l2g[i],this->Jacobian_g2l[TT(i,j)]);
                
            }
        }
    }
 
    

}

bool Mesh::inTriangle(Eigen::Vector2d q){
    if (q(0) >= 0 && q(0) <= 1 && q(1) >= 0 && q(1) <= 1 && (q(0)+q(1)) <=1 ){
        return true;
    }

    return false;
}