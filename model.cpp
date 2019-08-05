#include "model.h"
//#include "CellList.h"
#include "common.h"
#include <omp.h>
#include <unordered_map>
double const Model::T = 293.0;
double const Model::kb = 1.38e-23;
double const Model::vis = 1e-3;

extern Parameter parameter;

//#define OPENMP

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
    Bpp = parameter.Bpp * 1e9 * radius; //2.29 is Bpp/a/kT, now Bpp has unit of kT
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
    srand(parameter.seed);

    for(int i = 0; i < numP; i++){
        particles.push_back(particle_ptr(new Model::particle));
   }
    
}

#ifndef OPENMP
void Model::run() {
    if (this->timeCounter == 0 || ((this->timeCounter + 1) % trajOutputInterval == 0)) {
        this->outputTrajectory(this->trajOs);
    }

    calForces(); 
    for (int i = 0; i < numP; i++) {            

        for (int j = 0; j < 3; j++){
           particles[i]->vel(j) = diffusivity_t * particles[i]->F(j) + sqrt(2.0 * diffusivity_t/dt_) *(*rand_normal)(rand_generator);;
        }           
        this->moveOnMeshV2(i);
    }
    this->timeCounter++;
    
}
#else
void Model::run() {
    if (this->timeCounter == 0 || ((this->timeCounter + 1) % trajOutputInterval == 0)) {
        this->outputTrajectory(this->trajOs);
    }
    
    calForces();
    Eigen::MatrixXd randomDispMat;
    randomDispMat.setRandom(3,numP);
//    omp_set_num_threads(1);
#pragma omp parallel default(shared)
{//    std::cout << omp_get_num_threads() << std::endl;
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numP; i++) {            
//            Eigen::Vector3d velocity;
            for (int j = 0; j < 3; j++){
//                particles[i]->F(j) = 0.0;
                particles[i]->vel(j) = diffusivity_t * particles[i]->F(j) + sqrt(2.0 * diffusivity_t/dt_) * randomDispMat(j,i);
            }
//            particles[i]->vel = velocity;
            /* Obtain thread number */
//            int tid = omp_get_thread_num();
//            printf("Hello World from thread = %d\n", tid);
//            std::cout << omp_get_num_threads() << std::endl;
//            std::cout << numP << std::endl;
//            std::cout << velocity << std::endl;
            this->moveOnMesh(i);
        }

}    
//    this->moveOnMesh_OMP();
    this->timeCounter++;
    
}
#endif

void Model::run(int steps){
    for (int i = 0; i < steps; i++){
	run();
    }
}

void Model::diffusionStat(int steps){
    std::stringstream ss;
    ss << steps;
    this->trajOs.open(filetag + "Diffusion_" + ss.str() + ".txt");
    
    for (int n = 0; n < parameter.nCycles; n++) {
        this->readxyz(iniFile);
        std::cout << "model initialize at round " << n << std::endl;
        if (n == 0) {
                for (int i = 0; i < numP; i++) {
            std::cout << i << "\t";
            for (int j = 0; j < 2; j++){
                std::cout << particles[i]->local_r[j] << "\t";
            }
            std::cout << particles[i]->meshFaceIdx << "\t";
            for (int j = 0; j < 3; j++){
                std::cout << particles[i]->r(j) << "\t";
            }
            std::cout << std::endl;
        }
        
        
        }
        
        
        for (int i = 0; i < steps; i++) {
            
            calForces(); 
           for (int i = 0; i < numP; i++) {            

                for (int j = 0; j < 3; j++){
                    particles[i]->vel(j) = diffusivity_t * particles[i]->F(j) + sqrt(2.0 * diffusivity_t/dt_) *(*rand_normal)(rand_generator);
            }           
            this->moveOnMeshV2(i);    
        }
        
       
    }
        this->trajOs << particles[0]->meshFaceIdx << "\t";
                for (int j = 0; j < 3; j++){
            this->trajOs << particles[0]->r(j) << "\t";
        }
        
        this->trajOs << steps*this->dt_ << "\t";
        this->trajOs << std::endl;
    }
}


void Model::moveOnMesh_OMP(){
    
#pragma omp parallel
#pragma omp for schedule(dynamic)
    for (int p_idx = 0; p_idx < numP; p_idx++) {
        // first calculate the tangent velocity
        Eigen::Vector3d velocity = particles[p_idx]->vel;
        int meshIdx = this->particles[p_idx]->meshFaceIdx;
        Eigen::Vector3d normal = mesh->F_normals.row(meshIdx).transpose();
        Eigen::Vector3d tangentV = velocity - normal * (normal.dot(velocity));
        // local velocity representation
        Eigen::Vector2d localV = mesh->Jacobian_g2l[meshIdx] * tangentV;
        Eigen::Vector2d localQ_new;
        double t_residual = this->dt_;

        bool finishWrapFlag = false;
        double tol = 1e-8;
        int hitFlag = -1;
        int hitFlag_prev = -1;
        localQ_new = particles[p_idx]->local_r + localV*t_residual;
        int count = 0;
        while (!finishWrapFlag) {
            meshIdx = particles[p_idx]->meshFaceIdx;
            if (localQ_new(0) < 0.0 && hitFlag_prev != 2) {
                hitFlag = 2;
                hitFlag_prev = mesh->TTi(meshIdx, 2);
                //            std::cout << mesh->TTi.row(meshIdx) << std::endl;
            } else if (localQ_new(1) < 0.0 && hitFlag_prev != 0) {
                hitFlag = 0;
                hitFlag_prev = mesh->TTi(meshIdx, 0);
                //           std::cout << mesh->TTi.row(meshIdx) << std::endl;
            } else if ((localQ_new(0) + localQ_new(1) > 1.0) && (hitFlag_prev != 1)) {
                hitFlag = 1;
                hitFlag_prev = mesh->TTi(meshIdx, 1);
                //            std::cout << mesh->TTi << std::endl;
                //            std::cout << mesh->TT << std::endl;
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
            particles[p_idx]->meshFaceIdx = mesh->TT(meshIdx, hitFlag);

        }
        particles[p_idx]->r = mesh->coord_l2g[particles[p_idx]->meshFaceIdx](particles[p_idx]->local_r);
    }
}

void Model::moveOnMesh(int p_idx){
    // first calculate the tangent velocity
    Eigen::Vector3d velocity = particles[p_idx]->vel;
    int meshIdx = this->particles[p_idx]->meshFaceIdx;
    Eigen::Vector3d normal = mesh->F_normals.row(meshIdx).transpose();
    Eigen::Vector3d tangentV = velocity - normal*(normal.dot(velocity));
    // local velocity representation
    Eigen::Vector2d localV = mesh->Jacobian_g2l[meshIdx]*tangentV;
    Eigen::Vector2d localQ_new;
    double t_residual = this->dt_;

    bool finishWrapFlag = false;
    double tol = 1e-8;
    int hitFlag = -1;
    int hitFlag_prev = -1;
    localQ_new = particles[p_idx]->local_r + localV*t_residual;
    int count = 0;
    while (!finishWrapFlag){  
        meshIdx = particles[p_idx]->meshFaceIdx;
        if (localQ_new(0) < 0.0 && hitFlag_prev!=2){
            hitFlag = 2;
            hitFlag_prev = mesh->TTi(meshIdx,2);
//            std::cout << mesh->TTi.row(meshIdx) << std::endl;
        } else if (localQ_new(1) < 0.0 && hitFlag_prev!=0) {
            hitFlag = 0;
            hitFlag_prev = mesh->TTi(meshIdx,0);
 //           std::cout << mesh->TTi.row(meshIdx) << std::endl;
        } else if ((localQ_new(0) + localQ_new(1) > 1.0) && (hitFlag_prev!=1)){
            hitFlag = 1;
            hitFlag_prev = mesh->TTi(meshIdx,1);
//            std::cout << mesh->TTi << std::endl;
            
//            std::cout << mesh->TT << std::endl;
        } else {
            finishWrapFlag = true;
            particles[p_idx]->local_r = localQ_new;
            break;
        }
        count++;
        
        if (count >= 20){
            std::cerr << "inner loop interaction too long  " << count << "\t" << localV << std::endl;
        }
        localQ_new = mesh->localtransform_p2p[meshIdx][hitFlag](localQ_new);
        particles[p_idx]->meshFaceIdx = mesh->TT(meshIdx,hitFlag);
        
    }    
    particles[p_idx]->r = mesh->coord_l2g[particles[p_idx]->meshFaceIdx](particles[p_idx]->local_r);;
}  

void Model::moveOnMeshV3(int p_idx){
    // first calculate the tangent velocity
    Eigen::Vector3d velocity = particles[p_idx]->vel;
    int meshIdx = this->particles[p_idx]->meshFaceIdx;
    Eigen::Vector3d normal = mesh->F_normals.row(meshIdx).transpose();
    Eigen::Vector3d tangentV = velocity - normal*(normal.dot(velocity));
    // local velocity representation
    Eigen::Vector2d localV = mesh->Jacobian_g2l[meshIdx]*tangentV;
    Eigen::Vector2d localQ_new;
    double hitTime, hitIndex;
    double t_residual = this->dt_;
    // need to do wraping do ensure new particle position is finally lying on a surface
    while(t_residual > 1e-8){
        // move with local tangent speed
        localQ_new = particles[p_idx]->local_r + localV * t_residual;
        if (mesh->inTriangle(localQ_new)){
            t_residual = 0.0;
            // to avoid tiny negative number
            localQ_new(0) = abs(localQ_new(0));
            localQ_new(1) = abs(localQ_new(1));
            particles[p_idx]->local_r = localQ_new;
            break;
        } else {
            // if localQ_new(0) + localQ_new(1) > 1.0, then must hit edge 1
            // if localQ_new(0) < 0 and localQ_new(1) > 0, then must hit edge 2
            // if localQ_new(0) > 0 and localQ_new(1) < 0, then must hit edge 0
            // if localQ_new(0) < 0 and localQ_new(1) < 0, then we might hit edge 0 and 2
            // if localQ_new(0) > 1.0 and localQ_new(1
            Eigen::Vector3d t_hit;
            if ((localQ_new(0) + localQ_new(1) > 1.0) and localQ_new(0) > 0.0 and localQ_new(1) > 0.0) {
                hitTime = (1 - particles[p_idx]->local_r.sum()) / localV.sum();
                hitIndex = 1;
            } else if (localQ_new(0) < 0.0 and localQ_new(1) > 0.0) {
                hitTime = -particles[p_idx]->local_r(0) / localV(0);
                hitIndex = 2;
            } else if (localQ_new(1) < 0.0 and localQ_new(0) > 0.0) {
                hitTime = -particles[p_idx]->local_r(1) / localV(1);
                hitIndex = 0;
            } else if (localQ_new(1) < 0.0 and localQ_new(0) < 0.0) {
                t_hit(2) = -particles[p_idx]->local_r(0) / localV(0); // this second edge
                t_hit(0) = -particles[p_idx]->local_r(1) / localV(1); // the zero edge
                if (t_hit(0) < t_hit(2)){
                    hitTime = t_hit(0);
                    hitIndex = 0;
                } else {
                    hitTime = t_hit(2);
                    hitIndex = 2;
                }
            }
            
            
            t_residual -= hitTime;

            // here update the local coordinate
            particles[p_idx]->local_r += localV * hitTime;
            // here is the correction step to make simulation stable
            if( hitIndex == 0){
                // hit the first edge
                particles[p_idx]->local_r(1) = 0.0;                
            } else if( hitIndex == 1){
                // hit the second edge, local_r(0) + local_r(1) = 1.0
                particles[p_idx]->local_r(0) = 1.0 - particles[p_idx]->local_r(1);
            } else if (hitIndex == 2){
                // hit the third edge
                particles[p_idx]->local_r(0) = 0.0;
            }
            
            int meshIdx = particles[p_idx]->meshFaceIdx;



#ifdef DEBUG
            
            int reverseIdx = mesh->TTi(meshIdx, hitIndex);
            Eigen::Vector2d newV = mesh->localtransform_v2v[meshIdx][hitIndex]*localV;
            Eigen::Vector2d oldV = mesh->localtransform_v2v[mesh->TT(meshIdx,hitIndex)][reverseIdx]*newV;
            
            double diff1 = (localV - oldV).norm();
            
            //Eigen::Vector2d newr = mesh->localtransform_p2p[meshIdx][min_idx](particles[p_idx]->local_r);
            //Eigen::Vector2d oldr = mesh->localtransform_p2p[mesh->TT(meshIdx,min_idx)][reverseIdx](newr);
            
            //double diff2 = (oldr - particles[p_idx]->local_r).norm();
            
            if (diff1 > 1e-6) {
                std::cerr << this->timeCounter << " speed transformation error!" << std::endl;
            }
            
            //if (diff2 > 1e-6) {
            //    std::cerr << "position transformation error!" << std::endl;
            //}
            
            
            
#endif   
            // transform to the local tangent speed in the new plane
            localV = mesh->localtransform_v2v[meshIdx][hitIndex]*localV;

            // transform to local coordinate in the new plane
            // because the local coordinate is on the edge of the old plane; it also must be in the edge of new plane
            
            particles[p_idx]->local_r = mesh->localtransform_p2p[meshIdx][hitIndex](particles[p_idx]->local_r);
            if (abs(particles[p_idx]->local_r(0)) < 1e-8) {
                particles[p_idx]->local_r(0) = 2e-8;
            }
            if (abs(particles[p_idx]->local_r(1)) < 1e-8) {
                particles[p_idx]->local_r(1) = 2e-8;
            }
#ifdef DEBUG
            
            if (!mesh->inTriangle(particles[p_idx]->local_r)){
                std::cerr << this->timeCounter << " not in triangle!" << std::endl;
                std::cout <<  particles[p_idx]->local_r << std::endl;
            }
#endif      //      
            // particles[p_idx]->meshFaceIdx = mesh->TT(meshIdx,min_idx);
           }   
    }
    
    if (abs(particles[p_idx]->local_r(0)) < 1e-8) {
        particles[p_idx]->local_r(0) = 2e-8;
    }
    if (abs(particles[p_idx]->local_r(1)) < 1e-8) {
        particles[p_idx]->local_r(1) = 2e-8;
    }

    if (!mesh->inTriangle(particles[p_idx]->local_r)){
        std::cerr << this->timeCounter << " not in triangle!" << std::endl;
        std::cout <<  particles[p_idx]->local_r << std::endl;
    }
    
}


void Model::moveOnMeshV2(int p_idx){
    // first calculate the tangent velocity
    Eigen::Vector3d velocity = particles[p_idx]->vel;
    int meshIdx = this->particles[p_idx]->meshFaceIdx;
    Eigen::Vector3d normal = mesh->F_normals.row(meshIdx).transpose();
    Eigen::Vector3d tangentV = velocity - normal*(normal.dot(velocity));
    // local velocity representation
    Eigen::Vector2d localV = mesh->Jacobian_g2l[meshIdx]*tangentV;
    Eigen::Vector2d localQ_new;
    
    double t_residual = this->dt_;
    // the while loop will start with particle lying on the surface
    // need to do wraping do ensure new particle position is finally lying on a surface
    
    if (!mesh->inTriangle(particles[p_idx]->local_r)){
        std::cerr << this->timeCounter << " not in triangle before the loop!" << std::endl;
        std::cout <<  particles[p_idx]->local_r << std::endl;
    }
    double positionPrecision = 1e-8;
    
    while(t_residual > 1e-8){
        // move with local tangent speed
        localQ_new = particles[p_idx]->local_r + localV * t_residual;
        if (mesh->inTriangle(localQ_new)){
            t_residual = 0.0;
            // to avoid tiny negative number
            localQ_new(0) = abs(localQ_new(0));
            localQ_new(1) = abs(localQ_new(1));
            particles[p_idx]->local_r = localQ_new;
            break;
        } else {
            // if localQ_new(0) + localQ_new(1) > 1.0, then must hit edge 1
            // if localQ_new(0) < 0 and localQ_new(1) > 0, then must hit edge 2
            // if localQ_new(0) > 0 and localQ_new(1) < 0, then must hit edge 0
            // if localQ_new(0) < 0 and localQ_new(1) < 0, then we might hit edge 0 and 2
            
            
            // as long as the velocity vector is not parallel to the three edges, there will a collision
            Eigen::Vector3d t_hit;
            // t_hit will be negative if it move away from the edge
            t_hit(2) = -particles[p_idx]->local_r(0) / localV(0); // this second edge
            t_hit(0) = -particles[p_idx]->local_r(1) / localV(1); // the zero edge
            t_hit(1) = (1 - particles[p_idx]->local_r.sum()) / localV.sum(); // the first edge
            
            double t_min = t_residual;
            int min_idx = -1;
            // the least positive t_hit will hit
            for (int i = 0; i < 3; i++){
                if (t_hit(i) > 1e-12 && t_hit(i) <= t_min){
                    t_min = t_hit(i);
                    min_idx = i;
                }            
            }

            if (min_idx < 0){
                // based on above argument, at least one edge will be hitted
                std::cerr << this->timeCounter << "\t t_hit is not determined!" << std::endl;
                std::cerr << t_hit(0) << "\t" << t_hit(1) << "\t" << t_hit(2) << "\t tmin " << t_min << std::endl;
                std::cout <<  particles[p_idx]->local_r << std::endl;
                std::cout << localQ_new << std::endl;
                std::cout << localV << std::endl;
                
                break;
            }
            
            t_residual -= t_min;

            // here update the local coordinate
            particles[p_idx]->local_r += localV * t_min;
            // here is the correction step to make simulation stable
            if( min_idx == 0){
                // hit the first edge
                particles[p_idx]->local_r(1) = 0.0;                
            } else if( min_idx == 1){
                // hit the second edge, local_r(0) + local_r(1) = 1.0
                particles[p_idx]->local_r(0) = 1.0 - particles[p_idx]->local_r(1);
            } else{
                // hit the third edge
                particles[p_idx]->local_r(0) = 0.0;
            }
            
            int meshIdx = particles[p_idx]->meshFaceIdx;
            int newMeshIdx = mesh->TT(meshIdx, min_idx);


#ifdef DEBUG
            
            int reverseIdx = mesh->TTi(meshIdx,min_idx);
            Eigen::Vector2d newV = mesh->localtransform_v2v[meshIdx][min_idx]*localV;
            Eigen::Vector2d oldV = mesh->localtransform_v2v[mesh->TT(meshIdx,min_idx)][reverseIdx]*newV;
            
            double diff1 = (localV - oldV).norm();
            
          
            if (diff1 > 1e-6) {
                std::cerr << this->timeCounter << " speed transformation error! " << diff1 << std::endl;
            }
           
            
#endif   
            // transform to the local tangent speed in the new plane
            localV = mesh->localtransform_v2v[meshIdx][min_idx]*localV;

            // transform to local coordinate in the new plane
            // because the local coordinate is on the edge of the old plane; it also must be in the edge of new plane
            
            particles[p_idx]->local_r = mesh->localtransform_p2p[meshIdx][min_idx](particles[p_idx]->local_r);
            particles[p_idx]->meshFaceIdx = newMeshIdx;
            
            
            if (abs(particles[p_idx]->local_r(0)) < positionPrecision) {
                particles[p_idx]->local_r(0) = 2 * positionPrecision;
            } else if (abs(particles[p_idx]->local_r(1)) < positionPrecision) {
                particles[p_idx]->local_r(1) = 2 * positionPrecision;
            } else if (abs((1 - particles[p_idx]->local_r.sum())) < positionPrecision){
                particles[p_idx]->local_r(1) = 1.0 - positionPrecision - particles[p_idx]->local_r(0);
                particles[p_idx]->local_r(0) = 1.0 - positionPrecision - particles[p_idx]->local_r(1);
                
            } else {
                std::cerr << this->timeCounter << " not in triangle after wrapping!" << std::endl;
                std::cout <<  particles[p_idx]->local_r << std::endl;
            }
#ifdef DEBUG
            if (!mesh->inTriangle(particles[p_idx]->local_r)){
                std::cerr << "not in triangleafter wrapping and adjustment " << std::endl;
            }
#endif            

        }

    }



    if (!mesh->inTriangle(particles[p_idx]->local_r)){
        std::cerr << this->timeCounter << " not in triangle after the loop!" << std::endl;
        std::cout <<  particles[p_idx]->local_r << std::endl;
    }
    particles[p_idx]->r = mesh->coord_l2g[particles[p_idx]->meshFaceIdx](particles[p_idx]->local_r);;
}

/*    
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
#endif 
*/    

void Model::calForcesHelper(int i, int j, Eigen::Vector3d &F) {
    double dist;
    Eigen::Vector3d r;

    dist = 0.0;
    F.fill(0);
    r = particles[j]->r - particles[i]->r;
    dist = r.norm();
            
    if (dist < 2.0) {
        std::cerr << "overlap " << i << "\t" << j << "\t"<< this->timeCounter << "dist: " << dist << "\t" << this->timeCounter <<std::endl;
#ifdef OPENMP
        std::cerr << "report from thread: " << omp_get_thread_num() << std::endl;
        std::cerr << "number of threads: " << omp_get_num_threads() << std::endl; 
#endif
        dist = 2.06;
    }
    if (dist < cutoff) {
        // the unit of force is kg m s^-2
        // kappa here is kappa*a a non-dimensional number
        
        double Fpp = -4.0/3.0*
        Os_pressure*M_PI*(-3.0/4.0*pow(combinedSize,2.0)+3.0*dist*dist/16.0*radius_nm*radius_nm);
        Fpp = -Bpp * Kappa * exp(-Kappa*(dist-2.0));
//        Fpp += -9e-13 * exp(-kappa* (dist - 2.0));
        F = Fpp*r/dist;
    }
}

void Model::calForces() {
    
    
#ifndef OPENMP    
    for (int i = 0; i < numP; i++) {
        particles[i]->F.fill(0);
    }
    Eigen::Vector3d F;
    for (int i = 0; i < numP - 1; i++) {
        for (int j = i + 1; j < numP; j++) {
            calForcesHelper(i, j, F);
            particles[i]->F += F;
            particles[j]->F -= F;

        }
               
    }
    for (int i = 0; i < numP; i++) {
        particles[i]->F(2) -= parameter.fieldStrength;
    }
    
#else
#pragma omp parallel 
{
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numP; i++) {
        particles[i]->F.fill(0);
    }

#pragma omp barrier
    
#pragma omp for schedule(dynamic)    
    for (int i = 0; i < numP; i++) {
        for (int j = 0; j < numP; j++) {
            if (i!=j){
                Eigen::Vector3d F;
                calForcesHelper(i, j, F);
                particles[i]->F += F;
            }
        }
    }
    
}    

#endif
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
//        particles[i]->r = mesh->coord_l2g[particles[i]->meshFaceIdx](particles[i]->local_r); 
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
    
    // make sure that the initial configuration is not overlapped    
    
    for (int i = 0; i < numP - 1; i++){
        for (int j = i + 1; j < numP; j++){
            Eigen::Vector3d dist;
            dist = particles[i]->r - particles[j]->r;
            
            if (dist.norm() < parameter.cutoff) {
                std::cerr << "initial configuration too close!" << std::endl;
                
            }
            
        }
    }
    
}


void Model::generateConfig(){
    std::unordered_map<int,int> index_set;
    int count = 0; 
    // start generating initial config
    if (numP > mesh->numF){
        std::cerr << "generated N larger than Face number" << std::endl;
        exit(2);
    }
    
//    if (mesh->area_total / M_PI / 1.5 < numP){
//        std::cerr << "area too small" << std::endl;
//        exit(2);
    
    
//    }
    
    while(count < numP){
        int idx = rand() % mesh->numF;
        index_set[idx] = 1;
        count = index_set.size();
    }
    auto map_it = index_set.begin();
    for (int i = 0; i < numP; i++){
        particles[i]->meshFaceIdx = map_it->first;
        ++map_it;
        particles[i]->local_r(0) = 0.25;
        particles[i]->local_r(1) = 0.25;
       
        particles[i]->r = mesh->coord_l2g[particles[i]->meshFaceIdx](particles[i]->local_r);        
    }
    
    this->MCRelaxation();
     std::cout << "config generated!" << std::endl;
     
    std::ofstream os;
    std::stringstream ss;
    ss << numP;
    os.open("generatedConfig_" + ss.str() + ".txt");
    this->outputTrajectory(os);
    os.close();
    

}

void Model::MCRelaxation(){
    bool accept;
    bool notFinish = true;
    std::cout << "MC relaxation start!" << std::endl;
    int count = 0;
    double distCheck = 1;
    while (notFinish){
        std::cout << "step: " << count++ << std::endl;
        int overlapCount = 0;
        
        for (int i = 0; i < numP; i++){

            bool overlap_old = this->checkCloseness(i,distCheck, &accept);
            // if not overlap with other particle, then it is free
            particles[i]->free = !overlap_old;
            Eigen::Vector3d velocity;
            particles[i]->vel.setRandom(3,1);
            particles[i]->vel *= sqrt(mesh->area_avg)*0.25/dt_;
            int oldMeshIdx = particles[i]->meshFaceIdx;
            Eigen::Vector2d localQ_old = particles[i]->local_r;
            Eigen::Vector3d r_old = particles[i]->r;
            this->moveOnMesh(i);

            bool overlap = this->checkCloseness(i,distCheck,&accept);
            if(overlap_old){
                overlapCount++;
            }
    //        if (!overlap_old){

            // if initially is free, but now is overlapping, back to original coor
                if (particles[i]->free && overlap){
                    particles[i]->meshFaceIdx= oldMeshIdx;
                    particles[i]->local_r = localQ_old;
                    particles[i]->r = r_old;
                }
                if (!accept){
                // if initillay is not free, but now is overlapping with other free particle,back to original coor    
                    particles[i]->meshFaceIdx= oldMeshIdx;
                    particles[i]->local_r = localQ_old;
                    particles[i]->r = r_old;
                }
    //        }
        }
        std::cout << "overlap count: " << overlapCount << std::endl;
        if(overlapCount == 0){
            break;
        }
    }
    std::cout << "MC relaxation finish!" << std::endl;
}

bool Model::checkCloseness(int p_idx, double thresh, bool *accept){
    bool overlap = false;
    *accept = true;
    for (int i = 0; i < numP; i++){
        if (i != p_idx){
            Eigen::Vector3d dist;
            dist = particles[p_idx]->r - particles[i]->r;
            if (dist.norm() < thresh){
//                std::cout << "overlap: " << i << "\t" << p_idx <<"\t" << dist.norm() << std::endl;
                
                if (particles[i]->free){
                    *accept = false;
                }
                
                
                overlap = true;
            }
        
        }
    }
    return overlap;
}
