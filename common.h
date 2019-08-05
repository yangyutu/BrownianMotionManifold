#pragma once
#include <string>

struct Parameter {
    int N, dim, trajOutputInterval;
    double radius, dt, diffu_t, Bpp, Os_pressure, L_dep, cutoff, kappa;
    int numStep, nCycles;
    std::string iniConfig, filetag;
    int seed, PDE_nstep;
    std::string meshFile;
    double fieldStrength;
    double PDE_dt;
    int externalForce;
    double externalFz;
};



struct Parameter_cell:public Parameter {
    double tau,sigma,V_a,V_r,D0,beta;
    
};


class CoorPair {
public:
    int x;
    int y;

    CoorPair() {
    };

    CoorPair(int x0, int y0) {
        x = x0;
        y = y0;
    }

};

typedef struct {

    std::size_t operator()(const CoorPair & CP) const {
        std::size_t h1 = std::hash<int>()(CP.x);
        std::size_t h2 = std::hash<int>()(CP.y);
        return h1^(h2 << 1);
    }
} CoorPairHash;

typedef struct {

    bool operator()(const CoorPair & CP1, const CoorPair & CP2) const {
        return (CP1.x == CP2.x)&&(CP1.y == CP2.y);
    }
} CoorPairEqual;

