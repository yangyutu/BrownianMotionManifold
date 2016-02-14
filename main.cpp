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
#include "Mesh.h"
#include "common.h"
#include "model.h"

Parameter parameter;
void readParameter();
void testIO();
void testMyMesh();
void testModel();
int main(int argc, char *argv[])
{
//    testIO();
    testMyMesh();
//    testModel();
    return 0;
}

void testModel(){
    readParameter();
    Model m;
    m.mesh = std::make_shared<Mesh>();
    m.mesh->readMeshFile("shared/cube.off");
    m.mesh->initialize();
    m.createInitialState();
    
    for (int i = 0; i < 100; i++){
        m.run(100);    
    }

}


void testMyMesh(){
    Eigen::Vector3d v;
    std::cout << v << std::endl;
    
    Mesh mesh;
    mesh.readMeshFile("shared/cube.off");
    mesh.initialize();
    

}

void testIO() {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd F_normals;
    Eigen::MatrixXi E, EF, EI, TT, TTi;
    Eigen::MatrixXi uE;
    //Eigen::MatrixXi EMap;
    Eigen::VectorXi EMap;
    std::vector<std::vector<int>> uE2E;
    // Load a mesh in OFF format
    igl::readOFF("shared/2triangles.off", V, F);

    // Print the vertices and faces matrices
    std::cout << "Vertices: " << std::endl << V << std::endl;
    std::cout << "Faces:    " << std::endl << F << std::endl;
    std::cout << "num of Vertices: " << V.rows() << std::endl;
    std::cout << "num of faces: " << F.rows() << std::endl;

    std::vector<std::vector<int> > adj_list;

    igl::adjacency_list(F, adj_list);
    igl::all_edges(F, E);
    // E contains unordered directed edges
    std::cout << "directed EDGES: " << E << std::endl;
    //   EMAP #F*3 list of indices into uE, mapping each directed edge to unique
    //     undirected edge
    igl::unique_edge_map(F, E, uE, EMap, uE2E);



    // Inputs:
    //   F  #F by simplex_size list of mesh faces (must be triangles)
    // Outputs:
    //   TT   #F by #3 adjacent matrix, the element i,j is the id of the triangle adjacent to the j edge of triangle i
    //   TTi  #F by #3 adjacent matrix, the element i,j is the id of edge of the triangle TT(i,j) that is adjacent with triangle i
    // NOTE: the first edge of a triangle is [0,1] the second [1,2] and the third [2,3]. it is zero based.
    //       this convention is DIFFERENT from cotmatrix_entries.h
    igl::triangle_triangle_adjacency(F, TT, TTi);
    std::cout << "TT: " << std::endl << TT << std::endl;
    std::cout << "TTi:    " << std::endl << TTi << std::endl;



    igl::edge_flaps(F, E, EMap, EF, EI);

    // undirected edges does not contained repeated edges
    std::cout << "undirected Edges: " << uE << std::endl;
    std::cout << "Emap:  " << EMap << std::endl;
    std::cout << "uE to E" << std::endl;
    for (int i = 0; i < uE2E.size(); i++) {
        std::cout << "uE idx: " << i << std::endl;
        for (int j = 0; j < uE2E[i].size(); j++) {
            std::cout << uE2E[i][j] << std::endl;
        }

    }
    std::cout << "EF: " << std::endl << EF << std::endl;
    std::cout << "EI:    " << std::endl << EI << std::endl;

    for (int j = 0; j < adj_list.size(); j++) {
        std::cout << "vertex idx: " << j << std::endl;
        for (int i = 0; i < adj_list[j].size(); i++) {
            std::cout << adj_list[j][i] << std::endl;

        }
    }

    igl::per_face_normals(V, F, F_normals);
    std::cout << "Faces normals:    " << std::endl << F_normals << std::endl;

    // Save the mesh in OBJ format
    igl::writeOBJ("2triangles.obj", V, F);


}

void readParameter(){
    std::string line;
    std::ifstream runfile;
    runfile.open("run.txt");
    getline(runfile, line);
    runfile >> parameter.N;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.radius;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.nCycles;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.numStep;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.dt;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.diffu_t;    
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.Bpp;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.Os_pressure;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.L_dep;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.cutoff;   
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.kappa;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.seed;
    getline(runfile, line);
    getline(runfile, line);
    runfile >> parameter.trajOutputInterval;
    getline(runfile, line);
    getline(runfile, line);
    getline(runfile, parameter.iniConfig);
    getline(runfile, line);
    getline(runfile, parameter.filetag);
}
