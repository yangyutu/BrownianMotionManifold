#pragma once
#include<vector>
#include<memory>
#include<random>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Mesh.h"
#include "Eigen/Dense"
//#include "CellList.h"

//class CellList; 
//typedef std::shared_ptr<CellList> cellList_ptr;
class Model {
public:

    struct pos{
        double r[3];
        pos(double x = 0, double y = 0, double z = 0){
            r[0]=x;r[1]=y;r[2]=z;
        }
    };
    
    struct particle {
        Eigen::Vector3d r, F, vel;
        Eigen::Vector2d local_r;
        int meshFaceIdx;
        bool free;
        
        particle(){
            r.fill(0);
            local_r.fill(0);
            meshFaceIdx = 0;
        }
        
        particle(double x, double y, double z, int idx){
            r(0)=x;r(1)=y;r(2)=z;
            meshFaceIdx = idx;
        }
        
        particle(double q1,double q2, int idx){
            local_r(0) = q1;
            local_r(1) = q2;
            meshFaceIdx = idx;
        }
    };
    typedef std::shared_ptr<particle> particle_ptr;
    typedef std::vector<particle_ptr> state;
    typedef std::vector<std::shared_ptr<pos>> posArray;
   
    Model();
    virtual ~Model() {trajOs.close();
    opOs.close(); osTarget.close();
    }
    virtual void moveOnMesh(int p_idx);
    virtual void moveOnMeshV2(int p_idx);
    virtual void moveOnMeshV3(int p_idx);
    void moveOnMesh_OMP();
    void MCRelaxation();
    void diffusionStat(int steps);
    void generateConfig();
    bool checkCloseness(int p_idx,double thresh,bool* accept);
    virtual void run();
    virtual void run(int steps);
    virtual void createInitialState();
    virtual state getCurrState(){return particles;}
    int getDimP(){return dimP;}
    
    double dt(){return dt_;}
    int np(){return numP;}
    std::shared_ptr<Mesh> mesh;
    state particles;
protected:
    virtual void calForces();
    virtual void calForcesHelper(int i, int j, Eigen::Vector3d &F);
    int dimP;
    static const double kb, T, vis;
    int numP, numObstacles;
    double radius, radius_nm;
    double LJ,rm;
    double Bpp; //2.29 is Bpp/a/kT
    double Kappa; // here is kappa*radius
    double Os_pressure;
    double L_dep; // 0.2 of radius size, i.e. 200 nm
    double combinedSize;
    std::vector<double> velocity={0.0,2.0e-6,5.0e-6};
    
    posArray obstacles; 
    std::vector<int> control;
    std::string iniFile;
    double dt_, cutoff, mobility, diffusivity_r, diffusivity_t;
    std::default_random_engine rand_generator;
    std::shared_ptr<std::normal_distribution<double>> rand_normal;
    int trajOutputInterval;
    int timeCounter,fileCounter;
    std::ofstream trajOs, opOs, osTarget;
    std::string filetag;
    virtual void outputTrajectory(std::ostream& os);
    void readxyz(const std::string filename);
    
};


