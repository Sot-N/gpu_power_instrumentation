#ifndef NVMLIns
#define NVMLIns

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include "wrapper.h"


template <class T> class NVMLWriter 
{
  static unsigned long int count;
public:
  //member variables
  std::string mekongkernelname_;
  std::string mekongkernelfile_;
  bool InstrumentStatus = false;
  unsigned int DevCount = 0;
  char DevName[64]; 
  nvmlReturn_t Res;
  nvmlDevice_t Dev_ID;
  nvmlPciInfo_t Pci_Info;
  nvmlEnableState_t Pow_Mode;
  nvmlComputeMode_t Comp_Mode;
  pthread_t PowerInstrumentThread;
  
  //member functions:
  void InitKernelname(std::string mekongkernelname, std::string mekongkernelfile);
  unsigned long int set_counter();
  int warmup_kernel();
  void PowInstStart();
  void PowInstStop(std::string mekongkernelname, std::string mekongkernelfile, int counter);
  int NVMLError(nvmlReturn_t NVMLCheck);


//constructor
NVMLWriter(std::string mekongkernelname, std::string mekongkernelfile) {
          mekongkernelname_ = mekongkernelname;
          mekongkernelfile_ = mekongkernelfile;
}
//default constructor
NVMLWriter(){}


};
template <class T> unsigned long int NVMLWriter<T>::count = 0;
#endif
