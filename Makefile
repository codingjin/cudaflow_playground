# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -I taskflow/ --extended-lambda

# Targets
TARGETS = addmatmul cuda_add cuda_matmul cudaflow_matmul_timing matmul_timing stream flow

# Default target
all: $(TARGETS)

# Build rules
addmatmul: addmatmul.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

cuda_add: cuda_add.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

cuda_matmul: cuda_matmul.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

cudaflow_matmul_timing: cudaflow_matmul_timing.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

matmul_timing: matmul_timing.cu
	$(NVCC) -std=c++17 $< -o $@

stream: stream.cu
	$(NVCC) -std=c++17 $< -o $@

flow: flow.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

# Clean rule
clean:
	rm -f $(TARGETS) *.o

# Phony targets
.PHONY: all clean
