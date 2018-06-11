test:rmse.o leastsqr.o
	g++ -o test main.cpp recommend.cpp rmse.o leastsqr.o -L/usr/local/cuda-9.0/lib64 -I/usr/local/cuda-9.0/include -lcublas -lcuda -lcudart -Wextra -std=c++11

rmse.o: rmse.cu
	nvcc -std=c++11 -c -arch=sm_37 rmse.cu

leastsqr.o: leastsqr.cu
	nvcc -std=c++11 -c -arch=sm_37 leastsqr.cu

clean:
	rm -rf test *.o
