#include<iostream>
#include<utils/cmdline.hpp>

using std::cin;
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{

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

	int nthreads = std::min(Utils::gpu_count(),args.max_num_threads);
  	nthreads = std::max(1,nthreads);

	//COPY THE CODE FROM PSRGPU1 FOR DIFFERENT GPU IDS

	if (args.progress_bar)
    	cout << "Reading data from " << args.infilename.c_str()) << endl;
  
  	timers["reading"].start();
  	SigprocFilterbank filobj(filename);
  	timers["reading"].stop();
    
  	if (args.progress_bar)
	{
    		cout << "Complete (read time: " << timers["reading"].getTime()) << "s)" << endl;
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


