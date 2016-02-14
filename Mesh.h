#pragma once
#include<Eigen/Dense>
#include<iostream>
#include<string>

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

struct Mesh {
    int numV;
    int numF;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    // face adj information
    Eigen::MatrixXi TT, TTi;
    // face normals
    Eigen::MatrixXd F_normals;
    // store face edge information

    std::vector<Eigen::Matrix3d> Face_Edges;



    // each face should have its transformation matrix between local chart of R^3
    std::vector<CoordOp_g2l> coord_g2l;
    std::vector<CoordOp_l2g> coord_l2g;
    
    std::vector<std::vector<CoordOp_l2l>> localtransform_p2p;
    
    std::vector<std::vector<Eigen::Matrix2d>> localtransform_v2v;
    
    
    std::vector<Eigen::Matrix<double, 3, 2 >> Jacobian_l2g;
    std::vector<Eigen::Matrix<double, 2, 3 >> Jacobian_g2l;
    std::vector<Eigen::Vector3d> bases;
    std::vector<std::vector<Eigen::Matrix3d>> RotMat;

    void readMeshFile(std::string filename);
    void initialize();
    bool inTriangle(Eigen::Vector2d q);

};

