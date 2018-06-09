
all: recommend.o

recommend.o:
	nvcc -std=c++11 recommend.cu -lcublas -o rec
