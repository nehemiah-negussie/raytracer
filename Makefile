NVCC  = nvcc
FLAGS = -arch=sm_89 -std=c++17 -O2

all: voxel_rt

voxel_rt: main.cu output.cpp
	$(NVCC) $(FLAGS) -o voxel_rt main.cu output.cpp

clean:
	rm -f voxel_rt voxel_rt.exe output.png

