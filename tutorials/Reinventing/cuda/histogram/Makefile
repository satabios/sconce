NVCCFLAGS = -O3 -arch=sm_80

all: histogram

histogram: histogram.cu
	nvcc $(NVCCFLAGS) -o histogram histogram.cu

clean:
	rm -f histogram


run: all 
	./histogram $(MATRIX_SIZE) $(RUNS) $(CUDA_VISIABLE_DEVICE)
