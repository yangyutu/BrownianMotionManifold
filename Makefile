CC = gcc
CXX = g++

HOME=/home/yangyutu/
IGL_INCLUDE=-I./libigl/include

DEBUGFLAG=-DDEBUG -g3
RELEASEFLAG= -O3 
CXXFLAGS=  -std=c++0x $(BOOST_INCLUDE) $(IGL_INCLUDE)  -D__LINUX -fopenmp #-DOPENMP

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
	
