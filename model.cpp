#include "model.h"
//#include "CellList.h"
#include "common.h"

double const Model::T = 293.0;
double const Model::kb = 1.38e-23;
double const Model::vis = 1e-3;

extern Parameter parameter;

Model::Model(){
    rand_normal = std::make_shared<std::normal_distribution<double>>(0.0, 1.0);

   
    filetag = parameter.filetag;
    iniFile = parameter.iniConfig;
    numP = parameter.N;
    dimP = parameter.dim;
    radius = parameter.radius;
    dt_ = parameter.dt;
    diffusivity_t = parameter.diffu_t;// this corresponds the diffusivity of 1um particle
    diffusivity_t /= pow(radius,2); // diffusivity will be in unit a^2/s
    Bpp = parameter.Bpp * kb * T * 1e9; //2.29 is Bpp/a/kT
    Kappa = parameter.kappa; // here is kappa*radius
    Os_pressure = parameter.Os_pressure * kb * T * 1e9;
    L_dep = parameter.L_dep; // 0.2 of radius size, i.e. 200 nm
    radius_nm = radius*1e9;
    combinedSize = (1+L_dep)*radius_nm;
    mobility = diffusivity_t/kb/T;
    trajOutputInterval = parameter.trajOutputInterval;
    fileCounter = 0;
    cutoff = parameter.cutoff;
    this->rand_generator.seed(parameter.seed);

    for(int i = 0; i < numP; i++){
        particles.push_back(particle_ptr(new Model::particle));
   }
    
}

void Model::run() {
    if (this->timeCounter == 0 || ((this->timeCounter + 1) % trajOutputInterval == 0)) {
        this->outputTrajectory(this->trajOs);
    }

    calForces();

 
        for (int i = 0; i < numP; i++) {
            
            Eigen::Vector3d velocity;
            for (int j = 0; j < 3; j++){
//                velocity(j) = mobility * particles[i]->F(j) + sqrt(2.0 * diffusivity_t/dt_) * (*rand_normal)(rand_generator);
//                velocity(j) = 50.0 + 10*j;
            }
            velocity << 50,50,-50;
            
            
            this->moveOnMesh(i,velocity);
        }
    this->timeCounter++;
    
}

void Model::run(int steps){
    for (int i = 0; i < steps; i++){
	run();
    }
}

void Model::moveOnMesh(int p_idx, Eigen::Vector3d velocity){
    // first calculate the tangent velocity
    int meshIdx = this->particles[p_idx]->meshFaceIdx;
    Eigen::Vector3d normal = mesh->F_normals.row(meshIdx).transpose();
    Eigen::Vector3d tangentV = velocity - normal*(normal.dot(velocity));
    // local velocity representation
    Eigen::Vector2d localV = mesh->Jacobian_g2l[meshIdx]*tangentV;
    Eigen::Vector2d localQ_new;
    double t_residual = this->dt_;
    
    
    while(t_residual/dt_ > 1e-2){
        localQ_new = particles[p_idx]->local_r + localV*t_residual;
        if (mesh->inTriangle(localQ_new)){
            t_residual -= t_residual;
            particles[p_idx]->local_r = localQ_new;
            break;
        } else {
            // as long as the velocity vector is not parallel to the three edges, there will a collision
            Eigen::Vector3d t_hit;
            t_hit(2) = -particles[p_idx]->local_r(0)/localV(0); // this second edge
            t_hit(0) = -particles[p_idx]->local_r(1)/localV(1); // the zero edge
            t_hit(1) = (1-particles[p_idx]->local_r.sum())/localV.sum(); // the first edge
            
            double t_min = t_residual;
            int min_idx = -1;
            for (int i = 0; i < 3; i++){
                if (t_hit(i) >1e-8 && t_hit(i) <= t_min){
                    t_min = t_hit(i);
                    min_idx = i;
                }            
            }
            if (min_idx < 0){
                
                std::cerr << this->timeCounter << "\t t_hit is not determined!" << std::endl;
                break;
            }
            
            t_residual -= t_min;
            particles[p_idx]->local_r += localV*t_min;
            // here is the correction step to make simulation stable
            if( min_idx == 0){
                particles[p_idx]->local_r(1) = 0.0;
            } else if( min_idx == 1){
                particles[p_idx]->local_r(0) = 1.0 - particles[p_idx]->local_r(1);
            } else{
                particles[p_idx]->local_r(0) = 0.0;
            }
            
            
            int meshIdx = particles[p_idx]->meshFaceIdx;
#ifdef DEBUG
            
            int reverseIdx = mesh->TTi(meshIdx,min_idx);
            Eigen::Vector2d newV = mesh->localtransform_v2v[meshIdx][min_idx]*localV;
            Eigen::Vector2d oldV = mesh->localtransform_v2v[mesh->TT(meshIdx,min_idx)][reverseIdx]*newV;
            
            double diff1 = (localV - oldV).norm();
            
            Eigen::Vector2d newr = mesh->localtransform_p2p[meshIdx][min_idx](particles[p_idx]->local_r);
            Eigen::Vector2d oldr = mesh->localtransform_p2p[mesh->TT(meshIdx,min_idx)][reverseIdx](newr);
            
            double diff2 = (oldr - particles[p_idx]->local_r).norm();
            
            if (diff1 > 1e-6 || diff2 > 1e-6){
                std::cerr << "transformation error!" << std::endl;
            }
            
            
            
            
#endif   
            
            localV = mesh->localtransform_v2v[meshIdx][min_idx]*localV;
            particles[p_idx]->local_r = mesh->localtransform_p2p[meshIdx][min_idx](particles[p_idx]->local_r);
#ifdef DEBUG
            if (!mesh->inTriangle(particles[p_idx]->local_r)){
//                std::cerr << "not in triangle! " << std::endl;
            }
#endif            
            particles[p_idx]->meshFaceIdx = mesh->TT(meshIdx,min_idx);
        }   
    }
}


void Model::calForcesHelper(int i, int j, Eigen::Vector3d F) {
    double dist;
    Eigen::Vector3d r;

    dist = 0.0;
    F.fill(0);
    r = particles[j]->r - particles[j]->r;
    dist = r.norm();
            
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t" << j << "\t"<< this->timeCounter << "dist: " << dist <<std::endl;
        dist = 2.06;
    }
    if (dist < cutoff) {
//        double Fpp = LJ * (-12.0 * pow((rm / dist), 12) / dist + 12.0 * pow((rm / dist), 7) / dist);
        double Fpp = -4.0/3.0*
        Os_pressure*M_PI*(-3.0/4.0*pow(combinedSize,2.0)+3.0*dist*dist/16.0*radius_nm*radius_nm);
        Fpp += -Bpp * Kappa * exp(-Kappa*(dist-2.0));
//        Fpp += -9e-13 * exp(-kappa* (dist - 2.0));
        F = Fpp*r/dist;
    }
}

void Model::calForces() {
    Eigen::Vector3d F;
    for (int i = 0; i < numP; i++) {
        particles[i]->F.fill(0);
    }
    
    for (int i = 0; i < numP - 1; i++) {
        for (int j = i + 1; j < numP; j++) {
            calForcesHelper(i, j, F);
            particles[i]->F += F;
            particles[j]->F -= F;

        }
    }
        
}
    


void Model::createInitialState(){

    this->readxyz(iniFile);
    std::stringstream ss;
    std::cout << "model initialize at round " << fileCounter << std::endl;
    ss << this->fileCounter++;
    if (trajOs.is_open()) trajOs.close();
//    if (opOs.is_open()) opOs.close();

    this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
//    this->opOs.open(filetag + "op" + ss.str() + ".txt");
    this->timeCounter = 0;

}

void Model::outputTrajectory(std::ostream& os) {

    for (int i = 0; i < numP; i++) {
        os << i << "\t";
        for (int j = 0; j < 2; j++){
            os << particles[i]->local_r[j] << "\t";
        }
        os << particles[i]->meshFaceIdx << "\t";
        particles[i]->r = mesh->coord_l2g[particles[i]->meshFaceIdx](particles[i]->local_r); 
        for (int j = 0; j < 3; j++){
            os << particles[i]->r(j) << "\t";
        }
        
        os << this->timeCounter*this->dt_ << "\t";
        os << std::endl;
    }
}


void Model::readxyz(const std::string filename) {
    std::ifstream is;
    is.open(filename.c_str());
    std::string line;
    double dum;
    for (int i = 0; i < numP; i++) {
        getline(is, line);
        std::stringstream linestream(line);
        linestream >> dum;
        linestream >> particles[i]->local_r(0);
        linestream >> particles[i]->local_r(1);
        linestream >> particles[i]->meshFaceIdx;
        
    }
    
    for (int i = 0; i < numP; i++) {
        particles[i]->r = mesh->coord_l2g[particles[i]->meshFaceIdx](particles[i]->local_r);        
    }
    
    

    is.close();
}

