#include "Model_cell.h"
#include <cmath>

extern Parameter_cell parameter_cell;

Model_cell::Model_cell() {
    rand_normal = std::make_shared<std::normal_distribution<double>>(0.0, 1.0);


    filetag = parameter_cell.filetag;
    iniFile = parameter_cell.iniConfig;
    numP = parameter_cell.N;
    dt_ = parameter_cell.dt;

    D0 = parameter_cell.cutoff;

    trajOutputInterval = parameter_cell.trajOutputInterval;
    fileCounter = 0;
    tau = parameter_cell.tau;
    sigma = parameter_cell.sigma;
    V_a = parameter_cell.V_a;
    V_r = parameter_cell.V_r;
    beta = parameter_cell.beta;
    polar_vel = 1.0;
    this->rand_generator.seed(parameter_cell.seed);
    srand(parameter_cell.seed);

    for (int i = 0; i < numP; i++) {
        particles.push_back(particle_ptr(new Model::particle));
        cellInfo.push_back(CellInfo_ptr(new Model_cell::CellInfo));
        cellInfo[i]->polarization.fill(0);
    }
}

void Model_cell::run() {
    if (this->timeCounter == 0 || ((this->timeCounter + 1) % trajOutputInterval == 0)) {
        this->outputTrajectory(this->trajOs);
    }
    Eigen::MatrixXd randomDispMat;
    randomDispMat.setRandom(3, numP);
    
    calForces();
    for (int i = 0; i < numP; i++) {
//        cellInfo[i]->polarization.fill(-1);
//        int oldIdx = particles[i]->meshFaceIdx;
//                if (i == 146)
//            std::cerr << "polarization"<<cellInfo[i]->polarization << std::endl;
        particles[i]->vel = cellInfo[i]->polarization*polar_vel + particles[i]->F;
        cellInfo[i]->polarization *= (1 - dt_ / tau);
        cellInfo[i]->polarization += sqrt(2.0*sigma*dt_) * randomDispMat.col(i) + (beta * cellInfo[i]->inhibition)*dt_;

//            for (int j = 0; j < 3; j++){
//        if (std::isnan(cellInfo[i]->polarization[j])){
//            std::cerr << "particle vel NaN at step" << this->timeCounter << "\t" << i << std::endl;
//            std::cerr << particles[i]->r << std::endl;
//            std::cerr << particles[i]->vel << std::endl;
//            std::cerr << "polarization"<<cellInfo[i]->polarization << std::endl;
//             std::cerr <<"force" <<particles[i]->F << std::endl;
//             std::cerr << cellInfo[i]->inhibition << std::endl;
//             std::cerr << randomDispMat.col(i) << std::endl;
             
//            exit(1);
//        }
//        }
        if (!testOn2dFlag){
            this->moveOnMesh(i);
        } else{
            this->moveOn2d(i);
        }
    }
    this->timeCounter++;
}


void Model_cell::moveOn2d(int j){

    particles[j]->r += particles[j]->vel * dt_;
    particles[j]->r(2) = 0.0;
    if (periodicFlag){
        for (int i = 0; i < 2; i++){
            particles[j]->r(i) -= round(particles[j]->r(i)/box(i))*box(i);
        }
    }
}

void Model_cell::testOn2d(){
    std::ifstream is;
    is.open("2dconfig.txt");
    double dum;
    for (int i = 0; i < numP; i++){
        is >> dum;
        is >> particles[i]->r[0];
        is >> particles[i]->r[1];
        is >> particles[i]->r[2];
    }
    
    box(0) = 15;
    box(1) = 13.5;
    box(2) = 15;

    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    ss << this->fileCounter++;
    if (trajOs.is_open()) trajOs.close();
//    if (opOs.is_open()) opOs.close();

    this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
//    this->opOs.open(filetag + "op" + ss.str() + ".txt");
    this->timeCounter = 0;

    periodicFlag = 1;
    testOn2dFlag = 1;
    for(int i = 0; i < parameter_cell.numStep; i++){
        this->run();
    }


}

void Model_cell::calForcesHelper(int i, int j, Eigen::Vector3d &F, Eigen::Vector3d &inhib) {
    double dist;
    Eigen::Vector3d r;

    dist = 0.0;
    F.fill(0);
    inhib.fill(0);
    r = particles[j]->r - particles[i]->r;
    
    if (periodicFlag){
        for (int k = 0; k < 2; k++){
            r(k) -= round(r(k)/box(k))*box(k);
        }
    }
    
    dist = r.norm();
    double Fpp;

    if (dist < D0 && dist >= 1) {
        Fpp = V_a * (dist - 1) / (D0 - 1);
        F = Fpp * r / dist;
    } else if (dist < 1) {
        Fpp = -V_r * (1 - dist);
        F = Fpp * r / dist;
    } 

    if (dist < D0) {
        inhib = -r / dist *exp(-(dist-1.0));
    }
}

void Model_cell::calForces() {
    if (this->timeCounter == 0 || ((this->timeCounter + 1) % 100 == 0)) {
        this->buildNbList();
        std::cout << this->timeCounter << std::endl;
    }
    
    for (int i = 0; i < numP; i++) {
        particles[i]->F.fill(0);
        cellInfo[i]->inhibition.fill(0);
    }
    Eigen::Vector3d F, inhit;
    for (int i = 0; i < numP; i++) {
        int nbListSize = cellInfo[i]->nbList.size();
        for (int jj = 0; jj < nbListSize; jj++){
            int j = cellInfo[i]->nbList[jj];
            calForcesHelper(i, j, F, inhit);
            particles[i]->F += F;
//            particles[j]->F -= F;
            cellInfo[i]->inhibition += inhit;
//            cellInfo[j]->inhibition -= inhit;
        }
    }
}


void Model_cell::moveOnMesh(int p_idx) {
    // first calculate the tangent velocity
    Eigen::Vector3d velocity = particles[p_idx]->vel;
    int meshIdx = this->particles[p_idx]->meshFaceIdx;
    Eigen::Vector3d normal = mesh->F_normals.row(meshIdx).transpose();
    Eigen::Vector3d tangentV = velocity - normal * (normal.dot(velocity));
    // local velocity representation
    Eigen::Vector2d localV = mesh->Jacobian_g2l[meshIdx] * tangentV;
    // local polarization representation
    Eigen::Vector3d polarization = cellInfo[p_idx]->polarization;
    Eigen::Vector3d tangentPolar = polarization - normal * (normal.dot(polarization));

    Eigen::Vector2d localPolar = mesh->Jacobian_g2l[meshIdx] * tangentPolar;

    Eigen::Vector2d localQ_new, localPolar_new;
    double t_residual = this->dt_;

    bool finishWrapFlag = false;
    double tol = 1e-8;
    int hitFlag = -1;
    int hitFlag_prev = -1;
    localQ_new = particles[p_idx]->local_r + localV*this->dt_;
    localPolar_new = localPolar;
    int count = 0;
    while (!finishWrapFlag) {
        meshIdx = particles[p_idx]->meshFaceIdx;
        if (localQ_new(0) < 0.0 && hitFlag_prev != 2) {
            hitFlag = 2;
            hitFlag_prev = mesh->TTi(meshIdx, 2);
        } else if (localQ_new(1) < 0.0 && hitFlag_prev != 0) {
            hitFlag = 0;
            hitFlag_prev = mesh->TTi(meshIdx, 0);
        } else if ((localQ_new(0) + localQ_new(1) > 1.0) && (hitFlag_prev != 1)) {
            hitFlag = 1;
            hitFlag_prev = mesh->TTi(meshIdx, 1);

        } else {
            finishWrapFlag = true;
            particles[p_idx]->local_r = localQ_new;
            break;
        }
        count++;

        if (count >= 10) {
            std::cerr << "inner loop interaction too long  " << count << "\t" << localV << std::endl;
        }
        localQ_new = mesh->localtransform_p2p[meshIdx][hitFlag](localQ_new);
        // this is to parallel transport the polarization in local coorindate
        localPolar_new = mesh->localtransform_v2v[meshIdx][hitFlag]*(localPolar_new);
        particles[p_idx]->meshFaceIdx = mesh->TT(meshIdx, hitFlag);

    }
    particles[p_idx]->r = mesh->coord_l2g[particles[p_idx]->meshFaceIdx](particles[p_idx]->local_r);
    // this is to transform the local polarization to the global coordinate system
    cellInfo[p_idx]->polarization = mesh->Jacobian_l2g[particles[p_idx]->meshFaceIdx]*(localPolar_new);
    
//    for (int j = 0; j < 3; j++){
//        if (std::isnan(particles[p_idx]->r[j])){
//            std::cerr << "particle position NaN at step" << this->timeCounter << "\t" << p_idx << std::endl;
//            std::cerr << particles[p_idx]->r << std::endl;
//            std::cerr << particles[p_idx]->vel << std::endl;
//            std::cerr << cellInfo[p_idx]->polarization << std::endl;
//             std::cerr << particles[p_idx]->F << std::endl;
//            exit(1);
//        }
//    }
   
}

void Model_cell::buildNbList(){
    double dist;
    Eigen::Vector3d r;

    for (int i = 0; i <numP; i++){
        cellInfo[i]->nbList.clear();
    }
    for (int i = 0; i <numP-1; i++){
        for (int j=i+1;j<numP;j++){
        dist = 0.0;
        r = particles[j]->r - particles[i]->r;
        
        
        if (periodicFlag){
            for (int i = 0; i < 2; i++){
                r(i) -= round(r(i)/box(i))*box(i);
            }
        }
        dist = r.norm();
    
        if (dist < 2.0*D0 ) {
            cellInfo[i]->nbList.push_back(j);
            cellInfo[j]->nbList.push_back(i);
        }

        if (dist < 0.2*D0){
            std::cerr << "particles are too closed " << dist << std::endl;
        }
        }
    }
}

void Model_cell::outputTrajectory(std::ostream& os) {

    for (int i = 0; i < numP; i++) {
        os << i << "\t";
        for (int j = 0; j < 2; j++){
            os << particles[i]->local_r[j] << "\t";
        }
        os << particles[i]->meshFaceIdx << "\t";
       // particles[i]->r = mesh->coord_l2g[particles[i]->meshFaceIdx](particles[i]->local_r); 
        for (int j = 0; j < 3; j++){
            os << particles[i]->r(j) << "\t";
        }
        
        for (int j = 0; j < 3; j++){
            os << cellInfo[i]->polarization(j) << "\t";
        }
        
        os << this->timeCounter*this->dt_ << "\t";
        os << std::endl;
    }
}
