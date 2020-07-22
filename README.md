# gpu_power_instrumentation

# Description:
This program enables you to measure the power consumption of GPU CUDA kernels. Since most of the kernels are very short compared to the temporal resolution of the GPU powermeter, a loop of one second is inserted allowing adequate power measurements. If the kernel lasts longer than a second, then the kernel will not be repeated. Moreover, due to observed discrepancy between the power baseline before the first kernel and all other consecutive kernel executions, a warmup kernel is executed before every kernel to ensure the same power baseline.


# Requirements

* NVIDIA Management Library (NVML)
* Pthread library

# Example
1)Include into your CUDA implementation (test.cu) as headers the following files the **NVMLWriter.h**, **NVMLWriter.cpp** and **kernel.cu**

2)Include instrumentation as shown in the example below where a CUDA kernel with name 'test_kernel<<<b,t>>>()' and filename with 'test.cu'

```
std::stringstream mekong_kernelname0;
mekong_kernelname0<<"test_kernel";
std::stringstream mekong_kernelfile0;
mekong_kernelfile0<<"test";
NVMLWriter<double> nvml_test_kernel;
int nvml_test_kernel_counter = 0;
nvml_test_kernel.InitKernelname(mekong_kernelname0.str(), mekong_kernelfile0.str());
nvml_test_kernel.PowInstStart();//Starts the power measurement
for (auto start = std::chrono::steady_clock::now(), nvml_now = start; nvml_now < start + std::chrono::milliseconds{1000}; nvml_now std::chrono::steady_clock::now()){
	test_kernel<<<b,t>>>(); //CUDA kernel launch
	nvml_test_kernel_counter++;
	cudaDeviceSynchronize();
}
nvml_test_kernel.PowInstStop(mekong_kernelname0.str(), mekong_kernelfile0.str(), nvml_test_kernel_counter);//Stop the power measurements
```

3)Compile with -lpthread  -std=c++11 -lnvidia-ml 


# Output files: energy.csv, energy_counter.csv
The energy.csv contains the following information:
  
* kernelfile
* kernelname
* launch sequence
* measurement sequence
* time
* power


The energy_counter.csv contains the following information:
	
* kernelfile
* kernelname
* launch sequence inside the for loop (for_launc_seq)

Information from both files can be used for calculating the average execution time of each kernel (average_execution_time = time/for_launch_seq).
