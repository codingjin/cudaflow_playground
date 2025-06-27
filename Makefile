# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -I taskflow/ --extended-lambda

# Targets
TARGETS = addmatmul cuda_add cuda_matmul

# Default target
all: $(TARGETS)

# Build rules
addmatmul: addmatmul.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

cuda_add: cuda_add.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

cuda_matmul: cuda_matmul.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

# Clean rule
clean:
	rm -f $(TARGETS) *.o

# Phony targets
.PHONY: all clean