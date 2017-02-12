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
#include <sstream>

using std::cin;
using std::cout;
using std::endl;
using std::cerr;

#define POST_PROC 1

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
	double mean;
	double std_dev;
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
	cout << "Bin width = " << bin_width << "Hz" << endl;
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
	    //if (args.verbose)
	    std::cout << "Resampling to "<< acc_list[jj] << " m/s/s" << std::endl;
	    resampler.resample(d_tim,d_tim_r,size,acc_list[jj]);

	    //if (args.verbose)
	    std::cout << "Execute forward FFT" << std::endl;
	    r2cfft.execute(d_tim_r.get_data(),d_fseries.get_data());

	    //if (args.verbose)
	    std::cout << "Form interpolated power spectrum" << std::endl;
	    former.form_interpolated(d_fseries,pspec);

	    //if (args.verbose)
	    std::cout << "Normalise power spectrum" << std::endl;
	    stats::normalise(pspec.get_data(),mean*size,std*size,size/2+1);

	    //if (args.verbose)
	    std::cout << "Harmonic summing" << std::endl;
	    harm_folder.fold(pspec);

	    //if (args.verbose)
	    std::cout << "Finding peaks" << std::endl;
	    SpectrumCandidates trial_cands(tim.get_dm(),ii,acc_list[jj]);
	    cand_finder.find_candidates(pspec,trial_cands);
	    cand_finder.find_candidates(sums,trial_cands);

	    //if (args.verbose)
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
	timers["pulsar"]	= Stopwatch();
	timers["single_pulse"] 	= Stopwatch();
  	timers["total"].start();

	CmdLineOptions args;
	if (!read_cmdline_options(args,argc,argv))
    		ErrorChecker::throw_error("Failed to parse command line arguments.");

	int device_count;
	if( cudaSuccess != cudaGetDeviceCount(&device_count))
		// exits if there are no devices detected
		ErrorChecker::throw_error("There are no available CUDA-capable devices"); 

	cout << "There are " << device_count << " available devices" << endl;

	cudaDeviceProp properties;
	for(int i=0; i < device_count; i++)
  	{
   		cudaGetDeviceProperties(&properties, i);
        	cout << "Device " << i << ": " <<  properties.name << endl;
   	}

	cout << "Number of devices requested: " << args.max_num_threads << endl;
	if (device_count < args.max_num_threads)
		ErrorChecker::throw_error("The number of requested devices has to be lower or equal to the number of available devices");

	int nthreads = args.max_num_threads;

	if (!args.gpu_ids.empty())
	{
   		if (args.gpu_ids.size() != nthreads)
        	{
			cout << "The number of GPUs used must be the same as the number of IDs provided (if any)" << std::endl;
                	cout << "Will now terminate!" << endl;
                	return 1;
        	}
  	} else
  	{
		// will always tart with ID 0
   		for (int current_id = 0; current_id < nthreads; current_id++)
        	{
                	args.gpu_ids.push_back(current_id);
        	}
  	}


	args.verbose = true; // for testing purposes

	std::vector<int>::iterator ids_iterator;

 	cout << endl;

	cout << "Devices that will be used: " << endl;

	for (ids_iterator = args.gpu_ids.begin(); ids_iterator < args.gpu_ids.end(); ++ids_iterator)
  	{
   		cudaGetDeviceProperties(&properties, *ids_iterator);
        	cout << "Device " << *ids_iterator << ": " << properties.name << endl;
  	}


	if (args.verbose)
    		cout << "Using file: " << args.infilename << endl;

	std::string filename(args.infilename);

	if (args.progress_bar)
    		cout << "Reading data from " << args.infilename.c_str() << endl;

  	timers["reading"].start();

	unsigned int disp_diff = 10;		// so I don't have to make the whole thing again
	bool smooth = true;			// mean and stdev smoothing on/off

	cudaSetDevice(args.gpu_ids[0]);

  	SigprocFilterbank filobj(filename, disp_diff, smooth);
  	timers["reading"].stop();

  	if (args.progress_bar)
	{
    		cout << "Complete (read time: " << timers["reading"].getTime() << "s)" << endl;
  	}

	std::cout << "Starting dedispersion phase, common for both pulsar and single pulse detection" << std::endl;

  	Dedisperser dedisperser(filobj,args.gpu_ids,nthreads);	// dedisp_create_plan_multi invoked here
        if (args.killfilename!="")
        {
                if (args.verbose)
                        std::cout << "Using killfile: " << args.killfilename << std::endl;
                dedisperser.set_killmask(args.killfilename);
        }

    	std::cout << "Generating DM list" << std::endl;
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

	unsigned char *timeseries_data_ptr = trials.get_data();

	dedisp_plan original_plan = dedisperser.get_dedispersion_plan();

  	timers["dedispersion"].stop();

  	if (args.progress_bar)
	    	std::cout << "Dedispersion execution time: " << timers["dedispersion"].getTime() << "s\n";

	timers["pulsar"].start();

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

		cout << "Nthreads: " << nthreads << endl;

  		for (int ii=0;ii<nthreads;ii++){
    			workers[ii] = (new Worker(trials,dispenser,acc_plan,args,size,args.gpu_ids[ii]));
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

    		cout << "Distilling DMs" << endl;
  		dm_cands.cands = dm_still.distill(dm_cands.cands);
  		dm_cands.cands = harm_still.distill(dm_cands.cands);

		cout << "Running candidate scorer" << endl;
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
    				// fold_n checks if npdmp is smaller than the number of candidates
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
		stats.add_timing_info(timers);

		std::stringstream xml_filepath;
		xml_filepath << args.outdir << "/" << "pulsar_search_overview.xml";
		stats.to_file(xml_filepath.str());

		cout << "Finished pulsar searching\n";

                if(POST_PROC) {

                        cout << "Removing pulsar search lock" << endl;
                        rmdir("pulsar_lock");
                }

	}

	timers["pulsar"].stop();

	timers["single_pulse"].start();

	for(int ii = 0; ii < nthreads; ii++)
		cout << args.gpu_ids[ii];


	for(int ii = 1; ii < nthreads; ii++) {
		cout << ii << endl;
		cudaSetDevice(args.gpu_ids[ii]);
		cout << ii << endl;
		cudaDeviceReset();
		cout << ii << endl;
	}

	if( args.single_pulse_search || args.both_search )
	{

		cout << "Made it here" << endl;

		cudaSetDevice(args.gpu_ids[0]);
		cudaDeviceReset();
		std::cout << "Single pulse searching starts here\n";

		std::cout << "Heimdall, open the Bifrost!!\n";

		// because Bifrost opening Heimdall sounds wrong

		// create Heimdall pipeline object - use results from pre-peasoup dedispersion
		// don't really need the whole hd_create_pipeline in use as it only does the dedisp steps prior to the
		// dedispersion such as creating dm list etc.
		hd_params params;
		hd_set_default_params(&params);

		params.utc_start = filobj.get_utc_start();
		params.output_dir = args.outdir;
		params.verbosity = 3; // set the maximum verbosity level, for testing purposes
		params.sigproc_file = args.infilename;
		params.dm_min = args.dm_start;
		params.dm_max = args.dm_end;
		params.dm_tol = args.dm_tol;
		params.dm_pulse_width = args.dm_pulse_width;	// expected intrinsic pulse width
		params.dm_nbits = 8;				// number of bits per dedispersed sample
		params.use_scrunching = false;
		params.gpu_id = args.gpu_ids[0]; 		// need to work on this to enable multi-GPU support
		params.detect_thresh = 10.0;
		params.f0 = filobj.get_fch1();
		params.df = filobj.get_foff();
		params.dt = filobj.get_tsamp();
		params.nchans = filobj.get_nchans();
		//params.utc_start = filobj_get_utc_start();	// leave for now
		params.spectra_per_second = (double) 1.0/(double)params.dt;
		params.max_giant_rate = args.max_rate;

		// round nsamps_gulp to a nearest higher power of 2
		size_t power_two_gulp = 1 << (unsigned int)ceil(log2((double)args.gulp_size));
		params.nsamps_gulp = power_two_gulp;

		size_t nsamps_gulp = params.nsamps_gulp;
		float start_time = args.start_time;
		float read_time = args.read_time;

		// just in case someone puts negative time
		size_t start_time_samp = min((unsigned long long)0, (unsigned long long)ceil(start_time / params.dt));
		size_t read_time_samp = (size_t)ceil(read_time / params.dt);

		cout << start_time_samp << endl;

		// default behaviour - read everything
		if (read_time_samp == 0)
			read_time_samp = output_samps;

		cout << read_time_samp << endl;

		// make sure we process at least one full gulp
		// need to adjust start time
		read_time_samp = max((unsigned long long)nsamps_gulp, (unsigned long long)read_time_samp);
		start_time_samp = min((long long)start_time_samp, (long long)output_samps - (long long)nsamps_gulp);

		cout << "Will process " << read_time_samp << " starting at sample " << start_time_samp << endl;

		// check that we are not trying to read beyond what is available
		size_t end_time_samp = (size_t)min((unsigned long long)output_samps, (unsigned long long)start_time_samp + read_time_samp);

		size_t nbits = filobj.get_nbits();
		size_t stride = (params.nchans * nbits) / (8 * sizeof(char));

		size_t original_samples = output_samps;

		hd_pipeline pipeline;
		hd_error error;

		cout << "Will process " << read_time_samp << " samples, starting at sample " << start_time_samp << endl;

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

		// used to store the total number of samples processed so far
		size_t total_nsamps = 0;

		// move the starting point
		total_nsamps = start_time_samp;

  		size_t overlap = 0;
		bool stop_requested = false;

		// will stop execution when the number of samples is larger
		// or equal to output_samps

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
                       			total_nsamps, &nsamps_processed, timeseries_data_ptr, original_samples, args.both_search);

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

			if (total_nsamps + nsamps_processed > end_time_samp)
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

                if(POST_PROC) {

                        cout << "Removing single pulse search lock" << endl;
                        rmdir("single_lock");
                }


	} // end of the single pulse search if-statement

	timers["single_pulse"].stop();

	timers["total"].stop();

	cout << "Finished the program execution" << endl;


	// REMEMBER!! timers is a map!!
	cout << "Timing:" << endl
		<< "\t * reading the file " << timers["reading"].getTime() << endl
		<< "\t * dedispersion: " << timers["dedispersion"].getTime() << endl;
	if( args.pulsar_search || args.both_search)
		cout << "\t * pulsar search: " << timers["pulsar"].getTime() << endl;
	if( args.single_pulse_search || args.both_search )
		cout << "\t * single pulse search: " << timers["single_pulse"].getTime() << endl;

	return 0;
}
