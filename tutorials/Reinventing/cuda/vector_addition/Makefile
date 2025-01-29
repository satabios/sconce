NVCCFLAGS = -O3 -arch=sm_80

all: vector_addition

vector_addition: vector_addition.cu
	nvcc $(NVCCFLAGS) -o vector_addition vector_addition.cu
	
clean:
	rm -f vector_addition

run: all 
	./vector_addition