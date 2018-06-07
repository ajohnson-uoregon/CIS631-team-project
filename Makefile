
all: recommend.o

recommend.o:
	nvcc -c recommend.cu -lcublas -o rec
