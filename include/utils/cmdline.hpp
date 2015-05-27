#pragma once
#include <tclap/CmdLine.h>
#include <string>
#include <iostream>
#include <ctime>
#include <vector>

struct CmdLineOptions {
  std::string infilename;
  std::string outdir;
  std::string killfilename;
  std::string zapfilename;
  int max_num_threads;
  std::vector<int> gpu_ids;
  unsigned int size;
  float dm_start;
  float dm_end;
  float dm_tol;
  float dm_pulse_width;
  float acc_start;
  float acc_end;
  float acc_tol;
  float acc_pulse_width;
  float boundary_5_freq;
  float boundary_25_freq;
  float start_time;
  float read_time;
  int gulp_size;
  int nharmonics;
  int npdmp;
  int limit;
  float min_snr;
  float min_freq;
  int max_rate;
  float max_freq;
  int max_harm;
  float freq_tol;
  int verbose;
  bool progress_bar;
  //int dada_id; dada_id is hexadecimal, will look into TCLAP interpretation of hexes
  bool scrunch;
  bool pulsar_search;
  bool single_pulse_search;
  bool both_search;
};

std::string get_utc_str()
{
  char buf[128];
  std::time_t t = std::time(NULL);
  std::strftime(buf, 128, "./%Y-%m-%d-%H:%M_bifrost/", std::gmtime(&t));
  return std::string(buf);
}

bool read_cmdline_options(CmdLineOptions& args, int argc, char **argv)
{
  try
    {
      TCLAP::CmdLine cmd("Bifrost pipeline for pulsar and single pulse processing", ' ', "1.0");

      TCLAP::ValueArg<std::string> arg_infilename("f", "inputfile",
						  "File to process (.fil)",
                                                  true, "", "string", cmd);

      TCLAP::ValueArg<std::string> arg_outdir("o", "outdir",
					      "The output directory",
					      false, get_utc_str(), "string",cmd);
      
      TCLAP::ValueArg<std::string> arg_killfilename("k", "killfile",
						    "Channel mask file",
						    false, "", "string",cmd);

      TCLAP::ValueArg<std::string> arg_zapfilename("z", "zapfile",
                                                   "Birdie list file",
                                                   false, "", "string", cmd);

      TCLAP::ValueArg<int> arg_max_num_threads("t", "num_threads",
	                                            "The number of GPUs to use",
                                                 false, 3, "int", cmd);

      TCLAP::ValueArg<int> arg_limit("", "limit",
				     "upper limit on number of candidates to write out",
				     false, 1000, "int", cmd);

      TCLAP::ValueArg<size_t> arg_size("", "fft_size",
                                       "Transform size to use (defaults to lower power of two)",
                                       false, 0, "size_t", cmd);

      TCLAP::ValueArg<float> arg_dm_start("", "dm_start",
                                          "First DM to dedisperse to",
                                          false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_dm_end("", "dm_end",
                                        "Last DM to dedisperse to",
                                        false, 100.0, "float", cmd);

      TCLAP::ValueArg<float> arg_dm_tol("", "dm_tol",
                                        "DM smearing tolerance (1.11=10%)",
                                        false, 1.10, "float",cmd);

      TCLAP::ValueArg<float> arg_dm_pulse_width("", "dm_pulse_width",
                                                "Minimum pulse width for which dm_tol is valid",
                                                false, 64.0, "float (us)",cmd);

      TCLAP::ValueArg<float> arg_acc_start("", "acc_start",
					   "First acceleration to resample to",
					   false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_acc_end("", "acc_end",
					 "Last acceleration to resample to",
					 false, 0.0, "float", cmd);

      TCLAP::ValueArg<float> arg_acc_tol("", "acc_tol",
					 "Acceleration smearing tolerance (1.11=10%)",
					 false, 1.10, "float",cmd);

      TCLAP::ValueArg<float> arg_acc_pulse_width("", "acc_pulse_width",
                                                 "Minimum pulse width for which acc_tol is valid",
						 false, 64.0, "float (us)",cmd);

      TCLAP::ValueArg<float> arg_boundary_5_freq("", "boundary_5_freq",
                                                 "Frequency at which to switch from median5 to median25",
                                                 false, 0.05, "float", cmd);

      TCLAP::ValueArg<float> arg_boundary_25_freq("", "boundary_25_freq",
						  "Frequency at which to switch from median25 to median125",
						  false, 0.5, "float", cmd);

      TCLAP::ValueArg<int> arg_nharmonics("n", "nharmonics",
                                          "Number of harmonic sums to perform",
                                          false, 4, "int", cmd);

      TCLAP::ValueArg<int> arg_npdmp("", "npdmp",
                                     "Number of candidates to fold and pdmp",
                                     false, 0, "int", cmd);

      TCLAP::ValueArg<float> arg_min_snr("m", "min_snr",
                                         "The minimum S/N for a candidate",
                                         false, 9.0, "float",cmd);

      TCLAP::ValueArg<float> arg_min_freq("", "min_freq",
                                          "Lowest Fourier freqency to consider",
                                          false, 0.1, "float",cmd);

      TCLAP::ValueArg<float> arg_max_freq("", "max_freq",
                                          "Highest Fourier freqency to consider",
                                          false, 1100.0, "float",cmd);

	TCLAP::ValueArg<int> arg_max_rate("", "max_rate",
						"Maximum number of single pulse candidates per minute",
						false, 250000, "int", cmd);

      TCLAP::ValueArg<int> arg_max_harm("", "max_harm_match",
                                        "Maximum harmonic for related candidates",
                                        false, 16, "float",cmd);

      TCLAP::ValueArg<float> arg_freq_tol("", "freq_tol",
                                          "Tolerance for distilling frequencies (0.0001 = 0.01%)",
                                          false, 0.0001, "float",cmd);

	TCLAP::ValueArg<float> arg_start_time("", "start",
						"Time from the start of the observation to skip for the single pulse processing (s)",
						false, 0.0, "float", cmd);
	TCLAP::ValueArg<float> arg_read_time("", "read",
						"Time to read for the single pulse processing (s)",
						false, 0.0, "float", cmd);
	TCLAP::ValueArg<int> arg_gulp_size("", "gulp",
						"The number of time samples processed in one singple pulse search chunk",
						false, 262144, "int", cmd);

      TCLAP::MultiArg<int> arg_gpu_ids("i", "gpu_id", "GPU IDs to be used", false, "int", cmd);

      TCLAP::MultiSwitchArg arg_verbose("v", "verbose", "Verbose mode with different levels (up to 4)", cmd);

      //TCLAP::SwitchArg arg_verbose("v", "verbose", "verbose mode", cmd);

      TCLAP::SwitchArg arg_scrunch("s", "scrunch", "Enable scrunching", cmd);

      TCLAP::SwitchArg arg_progress_bar("p", "progress_bar", "Enable progress bar for DM search", cmd);

      TCLAP::SwitchArg arg_pulsar_search("", "pulsar", "Enable pulsar searching", cmd);

      TCLAP::SwitchArg arg_single_pulse_search("", "single", "Enable single pulse search", cmd);

      TCLAP::SwitchArg arg_both_search("", "both", "Enable search for both pulsar and single pulse search", cmd);

      cmd.parse(argc, argv);
      args.infilename        = arg_infilename.getValue();
      args.outdir            = arg_outdir.getValue();
      args.killfilename      = arg_killfilename.getValue();
      args.zapfilename       = arg_zapfilename.getValue();
      args.max_num_threads   = arg_max_num_threads.getValue();
      args.gpu_ids	     = arg_gpu_ids.getValue();
      args.limit             = arg_limit.getValue();
      args.size              = arg_size.getValue();
      args.dm_start          = arg_dm_start.getValue();
      args.dm_end            = arg_dm_end.getValue();
      args.dm_tol            = arg_dm_tol.getValue();
      args.dm_pulse_width    = arg_dm_pulse_width.getValue();
      args.acc_start         = arg_acc_start.getValue();
      args.acc_end           = arg_acc_end.getValue();
      args.acc_tol           = arg_acc_tol.getValue();
      args.acc_pulse_width   = arg_acc_pulse_width.getValue();
      args.boundary_5_freq   = arg_boundary_5_freq.getValue();
      args.boundary_25_freq  = arg_boundary_25_freq.getValue();
      args.nharmonics        = arg_nharmonics.getValue();
      args.npdmp             = arg_npdmp.getValue();
      args.min_snr           = arg_min_snr.getValue();
      args.min_freq          = arg_min_freq.getValue();
      args.max_freq          = arg_max_freq.getValue();
      args.max_harm          = arg_max_harm.getValue();
      args.max_rate		= arg_max_rate.getValue();
      args.freq_tol          = arg_freq_tol.getValue();
      args.verbose           = arg_verbose.getValue();
      args.scrunch	     = arg_scrunch.getValue();
      args.progress_bar      = arg_progress_bar.getValue();
      args.pulsar_search     = arg_pulsar_search.getValue();
      args.single_pulse_search = arg_single_pulse_search.getValue();
      args.both_search       = arg_both_search.getValue();
      args.start_time		= arg_start_time.getValue();
      args.read_time		= arg_read_time.getValue();
      args.gulp_size		= arg_gulp_size.getValue();
    }catch (TCLAP::ArgException &e) {
    std::cerr << "Error: " << e.error() << " for arg " << e.argId()
              << std::endl;
    return false;
  }
  return true;
}
