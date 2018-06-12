test:rmse.o leastsqr.o calculateLoss.o
	g++ -o test main.cpp recommend.cpp rmse.o leastsqr.o calculateLoss.o -L/usr/local/cuda-9.1/lib64 -I/usr/local/cuda-9.1/include -lcublas -lcuda -lcudart -lcurand -lcusolver -Wextra -std=c++11

rmse.o: rmse.cu
	nvcc -std=c++11 -c -arch=sm_50 rmse.cu

leastsqr.o: leastsqr.cu
	nvcc -std=c++11 -c -arch=sm_50 leastsqr.cu

calculateLoss.o: calculateLoss.cu
	nvcc -std=c++11 -c -arch=sm_50 calculateLoss.cu

clean:
	rm -rf test *.o
