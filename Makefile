CC = gcc
CXX = g++

HOME=/home/yuguangyang/
BOOST_INCLUDE=-I/opt/boost/boost_1_57_0
IGL_INCLUDE=-I/$(HOME)Dropbox/workspace/libigl/include

DEBUGFLAG=-DDEBUG -g3
RELEASEFLAG= -O3 -march=native -DARMA_NO_DEBUG 
CXXFLAGS=  -std=c++0x $(BOOST_INCLUDE) $(IGL_INCLUDE)  -D__LINUX -fopenmp -DDOPENMP

LDFLAG= -fopenmp -lpthread

OBJ=main.o Mesh.o model.o PDESolver.o Model_cell.o
all:test.exe 
test.exe: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAG) 
	
test_static: $(OBJ)
	$(CXX) -o $@ $^ -static $(LDFLAG) -lgomp -lm -ldl 
	
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $(RELEASEFLAG) $^
	


clean:
	rm *.o *.exe
	
