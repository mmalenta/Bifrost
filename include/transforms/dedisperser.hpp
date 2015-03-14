#pragma once
#include "dedisp.h"
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <data_types/timeseries.hpp>
#include <data_types/filterbank.hpp>
#include <utils/exceptions.hpp>

class Dedisperser {
private:
  dedisp_plan plan;
  Filterbank& filterbank;
  unsigned int num_gpus;
  std::vector<float> dm_list;
  std::vector<dedisp_bool> killmask;
  std::vector<int> gpu_ids;  

public:
  Dedisperser(Filterbank& filterbank, std::vector<int> gpu_ids, unsigned int num_gpus=1) // new dedisperser constructor using GPU IDs specified by the user
    :filterbank(filterbank), gpu_ids(gpu_ids), num_gpus(num_gpus)
  {
    killmask.resize(filterbank.get_nchans(),1);
    dedisp_error error = dedisp_create_plan_multi(&plan,			// dedisp_create_plan_multi is a part of dedisp library in file dedisp.cu
						  filterbank.get_nchans(),
						  filterbank.get_tsamp(),
						  filterbank.get_fch1(),
						  filterbank.get_foff(),
						  filterbank.get_mean(),
						  filterbank.get_std_dev(),
						  filterbank.get_mean_array(),
						  filterbank.get_stdev_array(),
						  filterbank.get_chunk_nsamps(),
						  num_gpus,
						  gpu_ids);
    ErrorChecker::check_dedisp_error(error,"create_plan_multi");
  }

  dedisp_plan  get_dedispersion_plan(void)
  {
	return plan;
  }

  void set_dm_list(float* dm_list_ptr, unsigned int ndms)
  {
    dm_list.resize(ndms);
    std::copy(dm_list_ptr, dm_list_ptr+ndms, dm_list.begin());
    dedisp_error error = dedisp_set_dm_list(plan,&dm_list[0],dm_list.size());
    ErrorChecker::check_dedisp_error(error,"set_dm_list");
  }

  void set_dm_list(std::vector<float> dm_list_vec)
  {
    dm_list.resize(dm_list_vec.size());
    std::copy(dm_list_vec.begin(), dm_list_vec.end(), dm_list.begin());
    dedisp_error error = dedisp_set_dm_list(plan,&dm_list[0],dm_list.size());
    ErrorChecker::check_dedisp_error(error,"set_dm_list");
  }

  std::vector<float> get_dm_list(void){
    return dm_list;
  }

  void generate_dm_list(float dm_start, float dm_end,
			float width, float tolerance)
  {
    dedisp_error error = dedisp_generate_dm_list(plan, dm_start, dm_end, width, tolerance);
    ErrorChecker::check_dedisp_error(error,"generate_dm_list");
    dm_list.resize(dedisp_get_dm_count(plan));
    const float* plan_dm_list = dedisp_get_dm_list(plan);
    std::copy(plan_dm_list,plan_dm_list+dm_list.size(),dm_list.begin());
  }

  void set_killmask(std::vector<int> killmask_in)
  {
    killmask.swap(killmask_in);
    dedisp_error error = dedisp_set_killmask(plan,&killmask[0]);
    ErrorChecker::check_dedisp_error(error,"set_killmask");
  }

  void set_killmask(std::string filename)
  {
    std::ifstream infile;
    std::string str;
    killmask.clear();
    infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile,filename);
    
    int ii=0;
    while(!infile.eof()&&ii<filterbank.get_nchans()){
      std::getline(infile, str);
      killmask.push_back(std::atoi(str.c_str()));
      ii++;
    }
    
    if (killmask.size() != filterbank.get_nchans()){
      std::cerr << "WARNING: killmask is not the same size as nchans" << std::endl;
      std::cerr << killmask.size() <<" != " <<  filterbank.get_nchans() <<  std::endl;
      killmask.resize(filterbank.get_nchans(),1);
    } else {
      dedisp_error error = dedisp_set_killmask(plan,&killmask[0]);
      ErrorChecker::check_dedisp_error(error,"set_killmask");
    }
    
  }
  
  //DispersionTrials<unsigned char> dedisperse(void);
  DispersionTrials<unsigned char> dedisperse(void)
  {
    size_t max_delay = dedisp_get_max_delay(plan);
    unsigned int out_nsamps = filterbank.get_nsamps()-max_delay;
    size_t output_size = out_nsamps * dm_list.size();
    unsigned char* data_ptr = new unsigned char [output_size]; 	// * 2 to handle 16-bit output data
    dedisp_error error = dedisp_execute(plan,
					filterbank.get_nsamps(),
					filterbank.get_data(),
					filterbank.get_nbits(),
					data_ptr,8,(unsigned)0);	// 8 is the number of output bits for dedispersed timeseries
    
    ErrorChecker::check_dedisp_error(error,"execute");
    DispersionTrials<unsigned char> ddata(data_ptr,out_nsamps,filterbank.get_tsamp(),dm_list);
    return ddata;
  }
};
