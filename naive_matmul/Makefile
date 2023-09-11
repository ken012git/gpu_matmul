CXX = g++
CXXFLAGS = -std=c++11 -Wall
LDFLAGS =
# LDCUDA = -L /usr/local/cuda/lib64  -L/usr/local/cuda/lib64/stubs/libcuda.so -lcudart -I /usr/local/cuda/include  
LDCUDA = -L/usr/local/cuda-11.1/lib64  -L/usr/local/cuda-11.1/lib64/stubs/libcuda.so -lcudart -I /usr/local/cuda-11.1/include  
CPPOBJ = main.o cpu_mm.o

NVCC = /usr/local/cuda-11.1/bin/nvcc
ARCH = sm_80
CUDAOBJ = gpu_cuda_mm.o

TARGET = main

all: $(TARGET)

# CUDAOBJ must follow LDCUDA, $^ follow the dependency order
$(TARGET): $(CPPOBJ) $(CUDAOBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDCUDA)

$(CPPOBJ): cpu_mm.hpp

$(CUDAOBJ): gpu_cuda_mm.hpp
	$(NVCC) -c -arch=$(ARCH) gpu_cuda_mm.cu

clean:
	rm -f $(CPPOBJ) $(CUDAOBJ) $(TARGET) *~
