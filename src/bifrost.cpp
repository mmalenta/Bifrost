#include<iostream>
#include<fstream>
#include<map>
#include<string>
#include<unistd.h>
#include<data_types/filterbank.hpp>
#include<transforms/dedisperser.hpp>
#include<utils/cmdline.hpp>
#include<utils/exceptions.hpp>
#include<utils/progress_bar.hpp>
#include"cuda.h"

using std::cin;
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{

  	std::map<std::string,Stopwatch> timers;
  	timers["reading"]      = Stopwatch();
  	timers["dedispersion"] = Stopwatch();
  	timers["searching"]    = Stopwatch();
  	timers["folding"]      = Stopwatch();
  	timers["total"]        = Stopwatch();
  	timers["total"].start();

	// MERGE PEASOUP AND HEIMDALL COMMAND LINE OPTIONS AND ADD ADITIONAL ONES
	// HEIMDALL USES SOME CUSTOM WRITTEN FUNCTION TO PARSE ARGUMENTS IN A CRUDE
	// WAY AS COMPARED TO PEASOUP WHICH MAKES USE OF TCLAP - MERGE DONE USING 
	// TCLAP FOR ALL COMMAND LINE OPTIONS	

	/*#####################################################################
	
	CHANGES:
		1. USE HEIMDALL'S -f FOR INPUT FILE
		2. REPLACE --id WITH -i FOR GPU ID CHOICE
		3. USE 4 DIFFERENT VERBOSITY LEVELS FROM HEIMDAL - USE MultiSwitchArg
		4. INCLUDED FROM HEIMDALL: -k (use PSRDADA hexadecimal key), -s
		(scrunching), - 	

	#####################################################################*/

	CmdLineOptions args;
	if (!read_cmdline_options(args,argc,argv))
    		ErrorChecker::throw_error("Failed to parse command line arguments.");

//	int nthreads = std::min(Utils::gpu_count(),args.max_num_threads);
//  	nthreads = std::max(1,nthreads);

	//COPY THE CODE FROM PSRGPU1 FOR DIFFERENT GPU IDS

	std::vector<int>::iterator iter_gpu_ids;

	// NEED TO CHECK IF THE NUMBER OF DEVICES IS HIGHER OR EQUAL TO THE NUMBER OF IDS SPECIFIED
	// NEED TO CHECK IF THE NUMBER OF IDS IS THE SAME AS THE NUMBER OF DEVICES SPECIFIED 
	// WITH -t COMMAND LINE OPTION (IF ANY)

	int device_count;
	if( cudaSuccess != cudaGetDeviceCount(&device_count))
		ErrorChecker::throw_error("There are no available CUDA-capable devices"); // exits if there are no devices detected
	
	cout << "There are " << device_count << " available devices" << endl;
	
	if (args.max_num_threads < args.gpu_ids.size() || device_count < args.gpu_ids.size() || args.max_num_threads < device_count )
		ErrorChecker::throw_error("The number of specified IDs must be lower than the number of GPUs available");

	if (args.gpu_ids.empty())
		for (int i = 0; i < args.max_num_threads; i++) args.gpu_ids.push_back(i);

	int nthreads = args.gpu_ids.size();

	cout << "Devices that will be used: " << endl;

	for (iter_gpu_ids = args.gpu_ids.begin(); iter_gpu_ids < args.gpu_ids.end(); ++iter_gpu_ids)
	{
		cudaDeviceProp device_properties;
		cudaGetDeviceProperties(&device_properties, *iter_gpu_ids);
		cout << device_properties.name << endl;
	}

	// MERGE PEASOUP AND HEIMDALL FILE READ FUNCTIONS

	if (args.verbose)
    		std::cout << "Using file: " << args.infilename << std::endl;
  	
	std::string filename(args.infilename);

	if (args.progress_bar)
    		cout << "Reading data from " << args.infilename.c_str() << endl;
  
  	timers["reading"].start();
  	SigprocFilterbank filobj(filename);
  	timers["reading"].stop();
    
  	if (args.progress_bar)
	{
    		cout << "Complete (read time: " << timers["reading"].getTime() << "s)" << endl;
  	}

  	Dedisperser dedisperser(filobj,nthreads);

  	if (args.killfilename!="")
	{
    		if (args.verbose)
      			std::cout << "Using killfile: " << args.killfilename << std::endl;
  		dedisperser.set_killmask(args.killfilename);
  	}
	
	return 0;
}


