#include <data_types/timeseries.hpp>
#include <data_types/fourierseries.hpp>
#include <data_types/candidates.hpp>
#include <data_types/filterbank.hpp>
#include <pipeline/error.hpp>
#include <pipeline/default_params.hpp>
#include <pipeline/pipeline_types.hpp>
#include <pipeline/pipeline.hpp>
#include <transforms/dedisperser.hpp>
#include <transforms/resampler.hpp>
#include <transforms/folder.hpp>
#include <transforms/ffter.hpp>
#include <transforms/dereddener.hpp>
#include <transforms/spectrumformer.hpp>
#include <transforms/birdiezapper.hpp>
#include <transforms/peakfinder.hpp>
#include <transforms/distiller.hpp>
#include <transforms/harmonicfolder.hpp>
#include <transforms/scorer.hpp>
#include <utils/mean_variance.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stats.hpp>
#include <utils/stopwatch.hpp>
#include <utils/progress_bar.hpp>
#include <utils/cmdline.hpp>
#include <utils/output_stats.hpp>
#include <string>
#include <iostream>
#include <fstream>                      // NEEDED FOR TEST OUTPUT FILES
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cufft.h"
#include "dedisp.h"
#include "pthread.h"
#include <cmath>
#include <map>

using std::cin;
using std::cout;
using std::endl;
using std::cerr;

struct dedisp_plan_struct {
  // Multi-GPU parameters 
  dedisp_size  device_count;
  // Size parameters
  dedisp_size  dm_count;
  dedisp_size  nchans;
  dedisp_size  max_delay;
  dedisp_size  gulp_size;
  // Physical parameters
  dedisp_float dt;
  dedisp_float f0;
  dedisp_float df;
  // Host arrays
  std::vector<dedisp_float> dm_list;      // size = dm_count
  std::vector<dedisp_float> delay_table;  // size = nchans
  std::vector<dedisp_bool>  killmask;     // size = nchans
  std::vector<dedisp_size>  scrunch_list; // size = dm_count
  // Device arrays //NEW: one for each GPU 
  std::vector< thrust::device_vector<dedisp_float> > d_dm_list;
  std::vector< thrust::device_vector<dedisp_float> > d_delay_table;
  std::vector< thrust::device_vector<dedisp_bool> >  d_killmask;
  std::vector< thrust::device_vector<dedisp_size> >  d_scrunch_list;
  //StreamType stream;
  // Scrunching parameters
  dedisp_bool  scrunching_enabled;
  dedisp_float pulse_width;
  dedisp_float scrunch_tol;
};

class DMDispenser {
private:
	DispersionTrials<unsigned char>& trials;
	pthread_mutex_t mutex;
	int dm_idx;
	int count;
	ProgressBar* progress;
	bool use_progress_bar;

public:
  	DMDispenser(DispersionTrials<unsigned char>& trials)
	:trials(trials),dm_idx(0),use_progress_bar(false){
    			count = trials.get_count();
    			pthread_mutex_init(&mutex, NULL);
  	}
  
  	void enable_progress_bar(){
    		progress = new ProgressBar();
    		use_progress_bar = true;
  	}

  	int get_dm_trial_idx(void){
    	pthread_mutex_lock(&mutex);
    	int retval;
    	if (dm_idx==0)
      		if (use_progress_bar){
			printf("Releasing DMs to workers...\n");
			progress->start();
      		}
    	if (dm_idx >= trials.get_count()){
      		retval =  -1;
      		if (use_progress_bar)
			progress->stop();
    	} else {
      		if (use_progress_bar)
			progress->set_progress((float)dm_idx/count);
      		retval = dm_idx;
      		dm_idx++;
    	}
    pthread_mutex_unlock(&mutex);
    return retval;
  }
  
  ~DMDispenser(){
    if (use_progress_bar)
      delete progress;
    pthread_mutex_destroy(&mutex);
  }
};

class Worker {
private:
  DispersionTrials<unsigned char>& trials;
  DMDispenser& manager;
  CmdLineOptions& args;
  AccelerationPlan& acc_plan;
  unsigned int size;
  int device;
  std::map<std::string,Stopwatch> timers;
  
public:
  CandidateCollection dm_trial_cands;

  Worker(DispersionTrials<unsigned char>& trials, DMDispenser& manager, 
	 AccelerationPlan& acc_plan, CmdLineOptions& args, unsigned int size, int device)
    :trials(trials),manager(manager),acc_plan(acc_plan),args(args),size(size),device(device){}
  
  void start(void)
  {
    //Generate some timer instances for benchmarking
    //timers["get_dm_trial"]      = Stopwatch();
    //timers["copy_to_device"] = Stopwatch();
    //timers["rednoise"]    = Stopwatch();
    //timers["search"]      = Stopwatch();

    cudaSetDevice(device);
    Stopwatch pass_timer;
    pass_timer.start();

    bool padding = false;
    if (size > trials.get_nsamps())
      padding = true;

    CuFFTerR2C r2cfft(size);
    CuFFTerC2R c2rfft(size);
    float tobs = size*trials.get_tsamp();
    float bin_width = 1.0/tobs;
    DeviceFourierSeries<cufftComplex> d_fseries(size/2+1,bin_width);
    DedispersedTimeSeries<unsigned char> tim;
    ReusableDeviceTimeSeries<float,unsigned char> d_tim(size);
    DeviceTimeSeries<float> d_tim_r(size);
    TimeDomainResampler resampler;
    DevicePowerSpectrum<float> pspec(d_fseries);
    Zapper* bzap;
    if (args.zapfilename!=""){
      if (args.verbose)
	std::cout << "Using zapfile: " << args.zapfilename << std::endl;
      bzap = new Zapper(args.zapfilename);
    }
    Dereddener rednoise(size/2+1);
    SpectrumFormer former;
    PeakFinder cand_finder(args.min_snr,args.min_freq,args.max_freq,size);
    HarmonicSums<float> sums(pspec,args.nharmonics);
    HarmonicFolder harm_folder(sums);
    std::vector<float> acc_list;
    HarmonicDistiller harm_finder(args.freq_tol,args.max_harm,false);
    AccelerationDistiller acc_still(tobs,args.freq_tol,true);
    float mean,std,rms;
    float padding_mean;
    int ii;

	PUSH_NVTX_RANGE("DM-Loop",0)
    while (true){
      //timers["get_trial_dm"].start();
      ii = manager.get_dm_trial_idx();
      //timers["get_trial_dm"].stop();

      if (ii==-1)
        break;
      trials.get_idx(ii,tim);

      if (args.verbose)
	std::cout << "Copying DM trial to device (DM: " << tim.get_dm() << ")"<< std::endl;

      d_tim.copy_from_host(tim);

      //timers["rednoise"].start()
      if (padding){
	    padding_mean = stats::mean<float>(d_tim.get_data(),trials.get_nsamps());
	    d_tim.fill(trials.get_nsamps(),d_tim.get_nsamps(),padding_mean);
      }

      if (args.verbose)
	    std::cout << "Generating acceleration list" << std::endl;
      acc_plan.generate_accel_list(tim.get_dm(),acc_list);

      if (args.verbose)
	    std::cout << "Searching "<< acc_list.size()<< " acceleration trials for DM "<< tim.get_dm() << std::endl;

      if (args.verbose)
	    std::cout << "Executing forward FFT" << std::endl;
      r2cfft.execute(d_tim.get_data(),d_fseries.get_data());

      if (args.verbose)
	    std::cout << "Forming power spectrum" << std::endl;
      former.form(d_fseries,pspec);

      if (args.verbose)
	    std::cout << "Finding running median" << std::endl;
      rednoise.calculate_median(pspec);

      if (args.verbose)
	    std::cout << "Dereddening Fourier series" << std::endl;
      rednoise.deredden(d_fseries);

      if (args.zapfilename!=""){
	    if (args.verbose)
	      std::cout << "Zapping birdies" << std::endl;
	    bzap->zap(d_fseries);
      }

      if (args.verbose)
	    std::cout << "Forming interpolated power spectrum" << std::endl;
      former.form_interpolated(d_fseries,pspec);

      if (args.verbose)
	    std::cout << "Finding statistics" << std::endl;
      stats::stats<float>(pspec.get_data(),size/2+1,&mean,&rms,&std);

      if (args.verbose)
	    std::cout << "Executing inverse FFT" << std::endl;
      c2rfft.execute(d_fseries.get_data(),d_tim.get_data());

      CandidateCollection accel_trial_cands;    
      PUSH_NVTX_RANGE("Acceleration-Loop",1)

      for (int jj=0;jj<acc_list.size();jj++){
	    if (args.verbose)
	      std::cout << "Resampling to "<< acc_list[jj] << " m/s/s" << std::endl;
	    resampler.resample(d_tim,d_tim_r,size,acc_list[jj]);

	    if (args.verbose)
	      std::cout << "Execute forward FFT" << std::endl;
	    r2cfft.execute(d_tim_r.get_data(),d_fseries.get_data());

	    if (args.verbose)
	      std::cout << "Form interpolated power spectrum" << std::endl;
	    former.form_interpolated(d_fseries,pspec);

	    if (args.verbose)
	      std::cout << "Normalise power spectrum" << std::endl;
	    stats::normalise(pspec.get_data(),mean*size,std*size,size/2+1);

	    if (args.verbose)
	      std::cout << "Harmonic summing" << std::endl;
	    harm_folder.fold(pspec);

	    if (args.verbose)
	      std::cout << "Finding peaks" << std::endl;
	    SpectrumCandidates trial_cands(tim.get_dm(),ii,acc_list[jj]);
	    cand_finder.find_candidates(pspec,trial_cands);
	    cand_finder.find_candidates(sums,trial_cands);

	    if (args.verbose)
	      std::cout << "Distilling harmonics" << std::endl;
	      accel_trial_cands.append(harm_finder.distill(trial_cands.cands));
      }
	  POP_NVTX_RANGE
      if (args.verbose)
	    std::cout << "Distilling accelerations" << std::endl;
      dm_trial_cands.append(acc_still.distill(accel_trial_cands.cands));
    }
	POP_NVTX_RANGE

    if (args.zapfilename!="")
      delete bzap;

    if (args.verbose)
      std::cout << "DM processing took " << pass_timer.getTime() << " seconds"<< std::endl;
  }

};

void* launch_worker_thread(void* ptr){
  reinterpret_cast<Worker*>(ptr)->start();
  return NULL;
}

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
		5. MAKE IT POSSIBLE TO CHOOSE BETWEEN PULSAR DETECTION, SINGLE PULSE DETECTION OR BOTH (DEFAULT)

	#####################################################################*/

	CmdLineOptions args;
	if (!read_cmdline_options(args,argc,argv))
    		ErrorChecker::throw_error("Failed to parse command line arguments.");

	//COPY THE CODE FROM PSRGPU1 FOR DIFFERENT GPU IDS - DONE

	std::vector<int>::iterator iter_gpu_ids;

	// NEED TO CHECK IF THE NUMBER OF DEVICES IS HIGHER OR EQUAL TO THE NUMBER OF IDS SPECIFIED
	// NEED TO CHECK IF THE NUMBER OF IDS IS THE SAME AS THE NUMBER OF DEVICES SPECIFIED 
	// WITH -t COMMAND LINE OPTION (IF ANY)
	// checks don't work if -t option not specified

	int device_count;
	if( cudaSuccess != cudaGetDeviceCount(&device_count))
		ErrorChecker::throw_error("There are no available CUDA-capable devices"); // exits if there are no devices detected

	cout << "There are " << device_count << " available devices" << endl;

	cudaDeviceProp properties;
	for(int i=0; i < device_count; i++)
  	{
   		cudaGetDeviceProperties(&properties, i);
        	std::cout << "Device " << i << ": " <<  properties.name << std::endl;
   	}

	cout << "Number of devices requested: " << args.max_num_threads << endl;
	cout << "Number of IDs entered: " << args.gpu_ids.size() << endl;
	if (args.max_num_threads < args.gpu_ids.size() || device_count < args.gpu_ids.size() || device_count < args.max_num_threads )
		ErrorChecker::throw_error("The number of specified IDs must be lower than the number of GPUs available");

	if (args.gpu_ids.empty())
		for (int i = 0; i < args.max_num_threads; i++) args.gpu_ids.push_back(i);

	int nthreads = args.gpu_ids.size();

	std::cout << "Number of GPUs that will be used: " << nthreads << std::endl;

	if (!args.gpu_ids.empty())
	{
   		if (args.gpu_ids.size() != nthreads) // just for testing - will later move to exceptions
        	{
			std::cout << "The number of GPUs used must be the same as the number of IDs provided (if any)" << std::endl;
                	std::cout << "Will now terminate!" << std::endl;
                	return 1;
        	}
  	} else
  	{
   		for (int current_id = 0; current_id < nthreads; current_id++)
        	{
                	args.gpu_ids.push_back(current_id);
        	}
  	}


	// ################# VERBOSE
	args.verbose = true; // for testing purposes

	std::vector<int>::iterator ids_iterator;

 	std::cout << std::endl;

	std::cout << "Devices that will be used: " << std::endl;

	for (ids_iterator = args.gpu_ids.begin(); ids_iterator < args.gpu_ids.end(); ++ids_iterator)
  	{
   		cudaGetDeviceProperties(&properties, *ids_iterator);
        	std::cout << "Device " << *ids_iterator << ": " << properties.name << std::endl;
  	}


	// MERGE PEASOUP AND HEIMDALL FILE READ FUNCTIONS

	if (args.verbose)
    		std::cout << "Using file: " << args.infilename << std::endl;

	std::string filename(args.infilename);

	if (args.progress_bar)
    		cout << "Reading data from " << args.infilename.c_str() << endl;

	// File reading is almost the same, subtle differences in header function

  	timers["reading"].start();
  	SigprocFilterbank filobj(filename);
  	timers["reading"].stop();

  	if (args.progress_bar)
	{
    		cout << "Complete (read time: " << timers["reading"].getTime() << "s)" << endl;
  	}

	std::cout << "Starting dedispersion phase, common for both pulsar and single pulse detection" << std::endl;

  	Dedisperser dedisperser(filobj,args.gpu_ids,nthreads);	// dedisp_create_plan_multi invoked here
	// Heimdall uses hd_create_pipeline to create dedispersion plan
	// what is called plan in peasoup is called pipeline in Heimdall - stick with the name plan
        if (args.killfilename!="")
        {
                if (args.verbose)
                        std::cout << "Using killfile: " << args.killfilename << std::endl;
                dedisperser.set_killmask(args.killfilename);
        }

    	std::cout << "Generating DM list" << std::endl;
	// I don't like the fact there are so many methods in the dedisperser class - use heimdall example of making it simpler later
  	dedisperser.generate_dm_list(args.dm_start,args.dm_end,args.dm_pulse_width,args.dm_tol);
  	std::vector<float> dm_list = dedisperser.get_dm_list();



	if (args.verbose)
	{
    		std::cout << dm_list.size() << " DM trials" << std::endl;
    		for (int ii=0;ii<dm_list.size();ii++)
      		std::cout << dm_list[ii] << std::endl;	// print out a list of DM trials
	}

	std::cout << "Executing dedispersion" << std::endl;

	if (args.progress_bar)
    	std::cout << "Starting dedispersion...\n";

  	timers["dedispersion"].start();
 	PUSH_NVTX_RANGE("Dedisperse",3)
  	DispersionTrials<unsigned char> trials = dedisperser.dedisperse();
  	POP_NVTX_RANGE

	size_t output_samps = trials.get_nsamps();

	size_t dm_size = trials.get_dm_list_size();

	size_t output_size = output_samps * dm_size;

	unsigned char *timeseries_data_ptr = new unsigned char [output_size];

	timeseries_data_ptr = trials.get_data();

	dedisp_plan original_plan = dedisperser.get_dedispersion_plan();

	cout << "dt: " << original_plan->dt << endl;

	// perform_tests(timeseries_data_ptr, output_samps, dm_size);

	cout << "Number of samples in the timeseries: " << output_samps << endl;
	cout << "Timeseries data size: " << output_size << endl;
	//cout << "Trials data pointer: " << trials.get_data();
	cout << "First data test: " << (int)timeseries_data_ptr[0] << " " << (int)timeseries_data_ptr[1] << endl;

  	timers["dedispersion"].stop();

  	if (args.progress_bar)
	    	std::cout << "Dedispersion execution time: " << timers["dedispersion"].getTime() << "s\n";

	if( args.pulsar_search || args.both_search)
	{

		std::cout << "Pulsar searching starts here\n";

		unsigned int size;
  		if (args.size==0)
    			size = Utils::prev_power_of_two(filobj.get_nsamps());
  		else
    			size = args.size;
  		if (args.verbose)
    			std::cout << "Setting transform length to " << size << " points" << std::endl;

  		AccelerationPlan acc_plan(args.acc_start, args.acc_end, args.acc_tol,
			    	args.acc_pulse_width, size, filobj.get_tsamp(),
			    	filobj.get_cfreq(), filobj.get_foff()); 


  		//Multithreading commands
  		timers["searching"].start();
  		std::vector<Worker*> workers(nthreads);
  		std::vector<pthread_t> threads(nthreads);
 		DMDispenser dispenser(trials);
  		if (args.progress_bar)
    			dispenser.enable_progress_bar();

  		for (int ii=0;ii<nthreads;ii++){
    			workers[ii] = (new Worker(trials,dispenser,acc_plan,args,size,ii));
    			pthread_create(&threads[ii], NULL, launch_worker_thread, (void*) workers[ii]);
  		}

  		DMDistiller dm_still(args.freq_tol,true);
  		HarmonicDistiller harm_still(args.freq_tol,args.max_harm,true,false);
  		CandidateCollection dm_cands;
  		for (int ii=0; ii<nthreads; ii++){
    			pthread_join(threads[ii],NULL);
    			dm_cands.append(workers[ii]->dm_trial_cands.cands);
  		}
  		timers["searching"].stop();

  		if (args.verbose)
    			std::cout << "Distilling DMs" << std::endl;
  		dm_cands.cands = dm_still.distill(dm_cands.cands);
  		dm_cands.cands = harm_still.distill(dm_cands.cands);

  		CandidateScorer cand_scorer(filobj.get_tsamp(),filobj.get_cfreq(), filobj.get_foff(),
			      	fabs(filobj.get_foff())*filobj.get_nchans());
 	 	cand_scorer.score_all(dm_cands.cands);

  		if (args.verbose)
    			std::cout << "Setting up time series folder" << std::endl;

  		MultiFolder folder(dm_cands.cands,trials);
  		timers["folding"].start();
  		if (args.progress_bar)
    			folder.enable_progress_bar();

  		if (args.npdmp > 0){
    			if (args.verbose)
      				std::cout << "Folding top "<< args.npdmp <<" cands" << std::endl;
    			folder.fold_n(args.npdmp);
  		}
  		timers["folding"].stop();

		if (args.verbose)
  			std::cout << "Writing output files" << std::endl;

		int new_size = std::min(args.limit,(int) dm_cands.cands.size());
		dm_cands.cands.resize(new_size);

		CandidateFileWriter cand_files(args.outdir);
		cand_files.write_binary(dm_cands.cands,"pulsar_candidates.peasoup");

		OutputFileWriter stats;
		stats.add_misc_info();
		stats.add_header(filename);
		stats.add_search_parameters(args);
		stats.add_dm_list(dm_list);

		std::vector<float> acc_list;
		acc_plan.generate_accel_list(0.0,acc_list);
		stats.add_acc_list(acc_list);

		stats.add_gpu_info(args.gpu_ids);
		stats.add_candidates(dm_cands.cands,cand_files.byte_mapping);
		timers["total"].stop();
		stats.add_timing_info(timers);

		std::stringstream xml_filepath;
		xml_filepath << args.outdir << "/" << "pulsar_search_overview.xml";
		stats.to_file(xml_filepath.str());

		std::cout << "Finished pulsar searching\n";

		//cudaDeviceReset();
	}

	if( args.single_pulse_search || args.both_search )
	{
		cudaDeviceReset();
		std::cout << "Single pulse searching starts here\n";

		std::cout << "Heimdall, open the Bifrost!!\n";
		// because Bifrost opening Heimdall sounds wrong

		cout << "Second data test: " << (int)timeseries_data_ptr[0] << " " << (int)timeseries_data_ptr[1] << endl;

		// create Heimdall pipeline object - use results from pre-peasoup dedispersion
		// don't really need the whole hd_create_pipeline in use as it only does the dedisp steps prior to the
		// dedispersion such as creating dm list etc.
		hd_params params;
		hd_set_default_params(&params);

		// copy command line options from args to params - due this ugly way now, put in the function later

		params.verbosity = 3; // set the maximum verbosity level, so we can have as much information as possible
		params.sigproc_file = args.infilename;
		params.dm_min = args.dm_start;
		params.dm_max = args.dm_end;
		params.dm_tol = args.dm_tol;
		params.dm_pulse_width = args.dm_pulse_width;	// expected intrinsic pulse width
		params.dm_nbits = 8;				// not sure what it does, but safer to keep same as input data
		params.use_scrunching =  false;
		params.gpu_id = 0; 				// need to work on this to enable multi-GPU support
		params.detect_thresh = 6.0;
		params.f0 = filobj.get_fch1();
		params.df = filobj.get_foff();
		params.dt = filobj.get_tsamp();
		params.nchans = filobj.get_nchans();
		//params.utc_start = filobj_get_utc_start();	// leave for now
		params.spectra_per_second = (double) 1.0/(double)params.dt;

		size_t nsamps_gulp = params.nsamps_gulp;
		size_t nbits = filobj.get_nbits();
		size_t stride = stride = (params.nchans * nbits) / (8 * sizeof(char));

		size_t original_samples = output_samps;

		hd_pipeline pipeline;
		hd_error error;

//		dedisp_plan original_plan = dedisperser.get_dedispersion_plan();

		cout << "dt: " << original_plan->dt << endl;

		//pipeline->set_dedispersion_plan(&original_plan);

		error = hd_create_pipeline(&pipeline, original_plan, params);

		if ( error != HD_NO_ERROR)
		{
			std::cerr << "ERROR: pipeline creation failed!!" << std::endl;
			return 1;
		}

		std::cout << "Pipeline created successfully!!" << std::endl;
		// hd_byte is unsigned char

		size_t total_nsamps = 0;

  		size_t overlap = 0;
		bool stop_requested = false;

		// will stop execution when the number of samples is larger 
		// or equal to output_samps - currently original_samples

		size_t nsamps_read = nsamps_gulp;

		while( nsamps_read && !stop_requested )
		{

    			if ( params.verbosity >= 1 )
			{
      				cout << "Executing pipeline on new gulp of " << nsamps_gulp
           				<< " samples..." << endl;
    			}

    			hd_size nsamps_processed = 0;

    			error = hd_execute(pipeline, nsamps_read+overlap, nbits,
                       			total_nsamps, &nsamps_processed, timeseries_data_ptr, original_samples);

    			if (error == HD_NO_ERROR)
    			{
      				if (params.verbosity >= 1)
       				cout << "Processed " << nsamps_processed << " samples." << endl;
    			}
    			else if (error == HD_TOO_MANY_EVENTS)
    			{
      				if (params.verbosity >= 1)
        				cerr << "WARNING: hd_execute produces too many events, some data skipped" << endl;
    			}
    			else
    			{
      				cerr << "ERROR: Pipeline execution failed" << endl;
      				cerr << "       " << hd_get_error_string(error) << endl;
      				hd_destroy_pipeline(pipeline);
      				return -1;
    			}

    			//pipeline_timer.stop();
    			//cout << "pipeline time: " << pipeline_timer.getTime() << " of " << (nsamps_read+overlap) * tsamp << endl;
    			//pipeline_timer.reset();

    			total_nsamps += nsamps_processed;

			cout << "Samples processed so far: " << total_nsamps << endl;

    			overlap += nsamps_read - nsamps_processed;

			if (total_nsamps + nsamps_processed > original_samples )
      				stop_requested = 1;

  		}

  		if( params.verbosity >= 1 )
		{
    			cout << "Successfully processed a total of " << total_nsamps
         			<< " samples." << endl;
  		}

  		if( params.verbosity >= 1 )
		{
    			cout << "Shutting down..." << endl;
  		}

  		hd_destroy_pipeline(pipeline);

  		if( params.verbosity >= 1 )
		{
	    		cout << "All done." << endl;
  		}

	} // end of the single pulse search if-statement

	cout << "Finished the program execution" << endl;

	return 0;
}


















