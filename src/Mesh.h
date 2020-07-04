#pragma once
#include<Eigen/Dense>
#include<iostream>
#include<string>
#include<vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
Coordinate transformation operator: local (2d) to global (3d)

Initialize the operator with one base vector (3d) and one Jacobian (3 by 2) cooresponding to the mesh triangle
*/

struct CoordOp_l2g {
    Eigen::Vector3d base;                       // 3D base coordinate
    Eigen::Matrix<double, 3, 2> Jacobian;       // Jacobian matrix for transforming a 2D local coordinate to a 3D global corrdinate

    CoordOp_l2g() {
    }

    CoordOp_l2g(const Eigen::Vector3d base0, const Eigen::Matrix<double, 3, 2> Jacobian0) {
        base = base0;
        Jacobian = Jacobian0;
    }

    Eigen::Vector3d operator()(const Eigen::Vector2d q_local) {
        return Jacobian * q_local + base;       // transformation from local to global is given by J * q_lobal + base
    }
};

/*
Coordinate transformation operator: global (3d) to local (2d)

Initialize the operator with one base vector (3d) and one Inverse Jacobian (2 by 3) cooresponding to the mesh triangle 

Note that from global to local will lose information if the global vector is not 
*/

struct CoordOp_g2l {
    Eigen::Vector3d base;                       // 3D base coordinate
    Eigen::Matrix<double, 2, 3> JacobianInv;    // Inverse Jacobian matrix for transforming a 3D global coordinate to a 2D local corrdinate

    CoordOp_g2l() {
    }

    CoordOp_g2l(const Eigen::Vector3d base0, const Eigen::Matrix<double, 2, 3> Jacobian0) {
        base = base0;
        JacobianInv = Jacobian0;
    }

    Eigen::Vector2d operator()(const Eigen::Vector3d q) {
        return JacobianInv * (q - base);        // transformation from global to local is given by J * q_lobal + base
    }
};

/*
Coordinate transformation operator: local (2d) to local (2d). This is particularly useful when a coordinate is at the common edge of two mesh triangles.

Initialize the operator with two base vectors (3d) and Jacobians cooresponding to the two mesh triangles 

The operator will take one local coordinate (2d) and output another local coordinate (2d)
*/

struct CoordOp_l2l {
    Eigen::Vector3d base1;
    Eigen::Vector3d base2;
    Eigen::Matrix<double, 3, 2> Jacobian1;
    Eigen::Matrix<double, 2, 3> JacobianInv2;
    CoordOp_l2l() {
    }

    CoordOp_l2l(const Eigen::Vector3d base10,const Eigen::Vector3d base20,
        const Eigen::Matrix<double, 3, 2> Jacobian10,
        const Eigen::Matrix<double, 2, 3> Jacobian20) {
        base1 = base10;
        base2 = base20;
        JacobianInv2 = Jacobian20;
        Jacobian1 = Jacobian10;
    }

    Eigen::Vector2d operator()(const Eigen::Vector2d q) {
        return JacobianInv2 * (Jacobian1*q + base1 - base2);    // The transformation is two steps: first convert to global, then convert to local again
    }
};


/*
Mesh class
A mesh is a collection of triangles. 
The mesh class contains the following information:
1) a vector of vertices 
2) a vector of normal vectors on the faces
3) a vector of coordinate transformation operators, including local to global, global to local

4) a vector of face adjacency information. Each face is guaranteed to have 3 neighbors.
5) a vector of inverse face adjacency information

6) Edge face information and convention

                 **
                 * *
     f:1, edge 1 *  *  f: 2, edge: 1
                 *   *
                 ******
                 f:0, edge:2
*/


struct Mesh {
    int numV;                                               // number of vertices
    int numF;                                               // number of faces
    double area_total;                                      // total area of triangle mesh
    double area_avg;                                        // average area of each triangle mesh
    Eigen::MatrixXd V;                                      // matrix storing vertice coordinates
    Eigen::MatrixXi F;                                      // matrix storing face information, every face is one row with three integers
    Eigen::MatrixXi TT, TTi;                                // face adjacency information
    Eigen::MatrixXd F_normals;                              // face normals
    Eigen::VectorXd dblA;                                   // store face edge information

    std::vector<Eigen::Matrix3d> Face_Edges;                // edge information for each faces

    
    std::vector<CoordOp_g2l> coord_g2l;                     // for each face, we have its transformation matrix from global to local
    std::vector<CoordOp_l2g> coord_l2g;                     // for each face, we have its transformation matrix from local to global
    
    std::vector<std::vector<CoordOp_l2l>> localtransform_p2p;               // for each face, we have its transformation matrix for coordinates from local to local
    std::vector<std::vector<Eigen::Matrix2d>> localtransform_v2v;           // for each face, we have its transformation matrix for velocities from local to local
    
    // Jacobians and their inverse, used for velocity transformation
    std::vector<Eigen::Matrix<double, 3, 2 >> Jacobian_l2g;                 // for each face, we have its Jacobian matrix from local to global, for velocity transformation
    std::vector<Eigen::Matrix<double, 2, 3 >> Jacobian_g2l;                 // for each face, we have its Inv Jacobian matrix from global to local, for velocity transformation
    std::vector<Eigen::Vector3d> bases;                                     // for each face, we store its base coordinate
    std::vector<std::vector<Eigen::Matrix3d>> RotMat;                       // for each face, we store its rotation matrix with respect to its neighboring faces

    // read mesh file
    void readMeshFile(std::string filename);                                
    
    // initialization
    void initialize();

    // check if a point is within the triangle
    bool inTriangle(Eigen::Vector2d q);

};