#pragma once
#include<Eigen/Dense>
#include<iostream>
#include<string>
#include<vector>

/*
Coordinate transformation operator: local (2d) to global (3d)

Initialize the operator with one base vector (3d) and one Jacobian (3 by 2) cooresponding to the mesh triangle
*/

struct CoordOp_l2g {
    Eigen::Vector3d base;
    Eigen::Matrix<double, 3, 2> Jacobian;

    CoordOp_l2g() {
    }

    CoordOp_l2g(const Eigen::Vector3d base0, const Eigen::Matrix<double, 3, 2> Jacobian0) {
        base = base0;
        Jacobian = Jacobian0;
    }

    Eigen::Vector3d operator()(const Eigen::Vector2d q_local) {
        return Jacobian * q_local + base;
    }
};

/*
Coordinate transformation operator: global (3d) to local (2d)

Initialize the operator with one base vector (3d) and one Inverse Jacobian (2 by 3) cooresponding to the mesh triangle 

Note that from global to local will lose information if the global vector is not 
*/

struct CoordOp_g2l {
    Eigen::Vector3d base;
    Eigen::Matrix<double, 2, 3> JacobianInv;

    CoordOp_g2l() {
    }

    CoordOp_g2l(const Eigen::Vector3d base0, const Eigen::Matrix<double, 2, 3> Jacobian0) {
        base = base0;
        JacobianInv = Jacobian0;
    }

    Eigen::Vector2d operator()(const Eigen::Vector3d q) {
        return JacobianInv * (q - base);
    }
};

/*
Coordinate transformation operator: local (2d) to local (2d)

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
        return JacobianInv2 * (Jacobian1*q + base1 - base2);
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
    int numV;
    int numF;
    double area_total,area_avg;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // face adj information
    Eigen::MatrixXi TT, TTi;
    // face normals
    Eigen::MatrixXd F_normals;
    // store face edge information
    Eigen::VectorXd dblA;

    std::vector<Eigen::Matrix3d> Face_Edges;

    // each face should have its transformation matrix between local chart of R^3
    std::vector<CoordOp_g2l> coord_g2l;
    std::vector<CoordOp_l2g> coord_l2g;
    
    std::vector<std::vector<CoordOp_l2l>> localtransform_p2p;
    
    std::vector<std::vector<Eigen::Matrix2d>> localtransform_v2v;
    
    // Jacobians and their inverse, used for speed transformation
    std::vector<Eigen::Matrix<double, 3, 2 >> Jacobian_l2g;
    std::vector<Eigen::Matrix<double, 2, 3 >> Jacobian_g2l;
    std::vector<Eigen::Vector3d> bases;
    std::vector<std::vector<Eigen::Matrix3d>> RotMat;

    void readMeshFile(std::string filename);
    void initialize();
    bool inTriangle(Eigen::Vector2d q);

};