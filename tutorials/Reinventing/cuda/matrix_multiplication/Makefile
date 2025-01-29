NVCCFLAGS = -O3 -arch=sm_80

all: naive_mm tiled_mm coarsened_tiled_mm

naive_mm: naive_mm.cu
	nvcc $(NVCCFLAGS) -o naive_mm naive_mm.cu

tiled_mm: tiled_mm.cu
	nvcc $(NVCCFLAGS) -o tiled_mm tiled_mm.cu

coarsened_tiled_mm: coarsened_tiled_mm.cu
	nvcc $(NVCCFLAGS) -o coarsened_tiled_mm coarsened_tiled_mm.cu
	
clean:
	rm -f naive_mm
	rm -f tiled_mm
	rm -f coarsened_tiled_mm

run: all 
	./naive_mm
	./tiled_mm
	./coarsened_tiled_mm

