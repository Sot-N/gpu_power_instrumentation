#include "NVMLWriter.h"
#include<cuda.h>
#include<stdio.h>

/*
Power instrumentation of GPU using NVML.
*/

//Load externally the kernel name using the kernelname variable
template <class T>  
void NVMLWriter<T>::InitKernelname(std::string mekongkernelname, std::string mekongkernelfile) {

    mekongkernelname_ = mekongkernelname; 
    mekongkernelfile_ = mekongkernelfile;

}


//This counter is tracking the number of launch. Basically, it counts the number of created objects(important for data merging)
template <class T>
unsigned long int NVMLWriter<T>::set_counter(){

    return count++;

}


//This function executes a warm-up kernel before profiling any kernel. It is necessary at least for the first kernel in each cuda file.
template <class T>
int NVMLWriter<T>::warmup_kernel(){

    //Call the wrapped kernel
    demo_kernel();
    return 0;

}


// Use p_thread to instrument the power consumption. This function needs as input a void pointer belongs to class
void *powerInstrument(void * tmp_ptr_)
{

    //This temporal pointer is casted therefore to be a pointer of class object. Otherwise this powerInstrument is not related to the class and there is an error.
    NVMLWriter<double> * writer_ptr_ = static_cast<NVMLWriter<double> *>(tmp_ptr_);
    unsigned int power = 0;
    //Create a file to append the data from each kernel execution, since both power and time is recorded the energy can be calculated
    FILE *f_energy = fopen("energy.csv", "a");
    //Create header row every time a new object is created.  
    fprintf(f_energy,"%s,%s,%s,%s,%s,%s\n","kernel_file","kernel_name","launch_seq","measurement_seq","time","power"); 
    //Call the member functions of this class object
    writer_ptr_->InitKernelname(writer_ptr_->mekongkernelname_, writer_ptr_->mekongkernelfile_);
    //Increment the object counter 
    unsigned long int seq_count = writer_ptr_->set_counter();
    //Assing the counter value in local variable avoiding re-counting the existed value by calling again the set_counter function from another object(kernel)
    unsigned long int local_seq_count = seq_count;
    //time_counter_ counts the number of power measurements or measurement_seq
    int time_counter_ = 0;
    //duration_ is how much time takes each individual power measurement. Set it to 0 for initializing it for each kernel 
    double duration_ = 0.0; 
    fprintf(f_energy,  "%s,%s,%lu,%d,%.lf, %.3d\n", writer_ptr_->mekongkernelfile_.c_str(), writer_ptr_->mekongkernelname_.c_str(), local_seq_count, time_counter_, duration_, 0);

        while (writer_ptr_->InstrumentStatus)
	{
            pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);

            //Extract power management mode of the GPU.
            writer_ptr_->Res = nvmlDeviceGetPowerManagementMode(writer_ptr_->Dev_ID,&(writer_ptr_->Pow_Mode));

            //Error checking
            writer_ptr_->NVMLError(writer_ptr_->Res);

            //Precise timer for measuring the starting time of each power measurement, needed for the duration
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

                //Check if power management mode is enabled.
                if (writer_ptr_->Pow_Mode == NVML_FEATURE_ENABLED)
                {
                    //The power consumption is given in milliWatts.
                    writer_ptr_->Res = nvmlDeviceGetPowerUsage(writer_ptr_->Dev_ID, &power);
                }
            
            //Precise time for measuring the ending time of each power measuremnt, needed for the duration
            std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	    
            //Duration of each power measurement. The given time is incrementally added.
            duration_ += std::chrono::duration_cast<std::chrono::microseconds>( stop - start ).count();
	    
            //Increment the counter inside the while loop, gives the measurement_seq
            time_counter_++;
		
            //Print necessary information csv file, convert the power into Watts
            fprintf(f_energy, "%s,%s,%lu,%d,%.lf,%.3lf\n", writer_ptr_->mekongkernelfile_.c_str(), writer_ptr_->mekongkernelname_.c_str(), local_seq_count, time_counter_, duration_, (power)/1000.0);

            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
	}
        // Close file
        fclose(f_energy);
        pthread_exit(0);
}


//Start power instrumentation
template <class T>  
void NVMLWriter<T>::PowInstStart()
{
    // Initialize NVML.
    Res = nvmlInit();
    if (NVML_SUCCESS != Res)
    {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(Res));
        exit(0);
    }

    // Check the number of connected GPUs.
    Res = nvmlDeviceGetCount(&DevCount);
    //printf("Device connected: %d\n", Res);

    if (NVML_SUCCESS != Res)
    {
        printf("Failed to identify connected GPU device: %s\n", nvmlErrorString(Res));
        exit(0);
    }

    for (int i = 0; i < DevCount; i++)
    {
        // Check the NVML ID, be carefull. This could be different than GPU id as given from nvidia driver/runtime.
        Res = nvmlDeviceGetHandleByIndex(i, &Dev_ID);
        if (NVML_SUCCESS != Res)
        {
            printf("Failed to handle the device %d: %s\n", i, nvmlErrorString(Res));
            exit(0);
        }

        // Check the device name.
        Res = nvmlDeviceGetName(Dev_ID, DevName, sizeof(DevName)/sizeof(DevName[0]));
        if (NVML_SUCCESS != Res)
        {
            printf("Failed to get the device name %d: %s\n", i, nvmlErrorString(Res));
            exit(0);
        }

        // Check device PCI information.
        Res = nvmlDeviceGetPciInfo(Dev_ID, &Pci_Info);
        if (NVML_SUCCESS != Res)
        {
            printf("Failed to get the device PCI info %d: %s\n", i, nvmlErrorString(Res));
            exit(0);
        }

        // Check the device compute mode.
        Res = nvmlDeviceGetComputeMode(Dev_ID, &Comp_Mode);
        if (NVML_ERROR_NOT_SUPPORTED == Res)
        {
            printf("Not a CUDA-capable device.\n");
        }
        else if (NVML_SUCCESS != Res)
        {
            printf("Failed to get the device compute mode %i: %s\n", i, nvmlErrorString(Res));
            exit(0);
        }
    }

    // The first index of NVML is used. However, this does not mean necessarily that the nvml ID is the same with cudasetDevice id of GPUs.This could be different for nvidia driver/runtime and nvidia-smi.
    Res = nvmlDeviceGetHandleByIndex(0, &Dev_ID);

    // Enable instrumentation 
    InstrumentStatus = true;
    //Call the warmup kernel before starting the power measurement
    int call = warmup_kernel();
    
    
    //Thread receives as arguments the function (powerInstrument) and as input to the function the pointer to the class object(this).
    int thread_err = pthread_create(&PowerInstrumentThread, NULL, powerInstrument, this);
    if (thread_err)
    {
        fprintf(stderr,"Error - pthread_create() return code: %d\n",thread_err);
        exit(0);
    }
}


// Stop power instrumentation 
template <class T>  
void NVMLWriter<T>::PowInstStop(std::string mekongkernelname, std::string mekongkernelfile, int counter)
{

    cudaDeviceSynchronize();
    InstrumentStatus = false;
    pthread_join(PowerInstrumentThread, NULL);

    //Shutdown the nvml
    Res = nvmlShutdown();
    if (NVML_SUCCESS != Res)
    {
        printf("Failed to shut down NVML: %s\n", nvmlErrorString(Res));
        exit(0);
    }

    mekongkernelname_ = mekongkernelname;
    mekongkernelfile_ = mekongkernelfile;

    // Create a file where you save how many time the kernel is executed when for loops are inserted. This allows to derive the execution time for short kernels
    FILE *f_energy_counter = fopen("energy_counter.csv", "a");
 
    // Create the first header
    fprintf(f_energy_counter,"%s,%s,%s\n","kernel_file","kernel_name","for_launch_seq");

    // Write the information after finish the kernel
    fprintf(f_energy_counter,"%s,%s,%d\n", mekongkernelfile_.c_str(), mekongkernelname_.c_str(), counter);

    // Close file
    fclose(f_energy_counter);
}

/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
template <class T>  
int NVMLWriter<T>::NVMLError(nvmlReturn_t NVMLCheck)
{

    if (NVMLCheck == NVML_ERROR_UNINITIALIZED)
        return 1;
    if (NVMLCheck == NVML_ERROR_INVALID_ARGUMENT)
        return 2;
    if (NVMLCheck == NVML_ERROR_NOT_SUPPORTED)
        return 3;
    if (NVMLCheck == NVML_ERROR_NO_PERMISSION)
        return 4;
    if (NVMLCheck == NVML_ERROR_ALREADY_INITIALIZED)
        return 5;
    if (NVMLCheck == NVML_ERROR_NOT_FOUND)
        return 6;
    if (NVMLCheck == NVML_ERROR_INSUFFICIENT_SIZE)
        return 7;
    if (NVMLCheck == NVML_ERROR_INSUFFICIENT_POWER)
        return 8;
    if (NVMLCheck == NVML_ERROR_DRIVER_NOT_LOADED)
        return 9;
    if (NVMLCheck == NVML_ERROR_TIMEOUT)
        return 10;
    if (NVMLCheck == NVML_ERROR_IRQ_ISSUE)
        return 11;
    if (NVMLCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
        return 12;
    if (NVMLCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
        return 13;
    if (NVMLCheck == NVML_ERROR_CORRUPTED_INFOROM)
        return 14;
    if (NVMLCheck == NVML_ERROR_GPU_IS_LOST)
        return 15;
    if (NVMLCheck == NVML_ERROR_RESET_REQUIRED)
        return 16;
    if (NVMLCheck == NVML_ERROR_OPERATING_SYSTEM)
        return 17;
    if (NVMLCheck == NVML_ERROR_LIB_RM_VERSION_MISMATCH)
        return 18;
    if (NVMLCheck == NVML_ERROR_IN_USE)
        return 19;
    if (NVMLCheck == NVML_ERROR_MEMORY)
        return 20;
    if (NVMLCheck == NVML_ERROR_NO_DATA)
        return 21;
    if (NVMLCheck == NVML_ERROR_VGPU_ECC_NOT_SUPPORTED)
        return 22;
    if (NVMLCheck == NVML_ERROR_UNKNOWN)
        return 999;

    return 0;
}

//set the type of template will be used. Otherwise, the compile does not know what kind of class object has been generated
template class NVMLWriter <double>; 
//template class NVMLWriter <float>; 
