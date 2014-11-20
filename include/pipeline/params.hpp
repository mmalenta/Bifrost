/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include "pipeline/pipeline_types.hpp"
#include <string>
#include <time.h>


// TODO: Consider grouping these into sub structs
struct hd_params {
  // Application parameters
  int      verbosity;
  std::string  sigproc_file;  // Name of sigproc filterbank file
  bool     yield_cpu;      // Yield/spin the CPU to in/decrease GPU latency
  hd_size  nsamps_gulp;    // No. samples to gulp into memory and process at once
  hd_size  dm_gulp_size;   // No. DMs to compute at once
  // Normalisation parameters
  hd_float baseline_length; // No. seconds over which to smooth the baseline
  // Observational parameters
  hd_size  beam;           // Beam index (0-based)
  bool     override_beam;  // override the beam in the file
  hd_size  nchans;         // No. frequency channels
  hd_float dt;             // Sampling time
  hd_float f0;             // Frequency of channel 0
  hd_float df;             // Frequency difference between channels
  // Dedispersion parameters
  hd_float dm_min;         // Min DM to search from
  hd_float dm_max;         // Max DM to search to
  hd_float dm_tol;         // Smear tolerance factor between DM trials
  hd_float dm_pulse_width; // Expected intrinsic pulse width
  hd_size  dm_nbits;       // No. bits per sample for dedispersed time series
  bool     use_scrunching; // Whether to apply time-scrunching during dedispersion
  hd_float scrunch_tol;    // Smear tolerance factor for time scrunching
  // RFI mitigation parameters
  hd_float rfi_tol;        // Probability of incorrectly identifying noise as RFI
  hd_size  rfi_min_beams;  // Min no. beams to identify coincident signals as RFI
  // Single pulse search parameters
  hd_size  boxcar_max;     // Max boxcar width to convolve with
  hd_float detect_thresh;  // Detection threshold (units of std. dev.)
  
  hd_size  cand_sep_time;   // Min separation between candidates (in samples)
  hd_size  cand_sep_filter; // Min separation between candidates (in filters)
  hd_size  cand_sep_dm;     // Min separation between candidates (in DM trials)
  hd_size  cand_rfi_dm_cut; // Minimum DM for valid candidate
  //hd_size  cand_min_members; // Minimum members for valid candidate
  
  hd_float max_giant_rate; // Maximum allowed number of giants per minute
  hd_size  min_tscrunch_width; // Filter width at which to begin tscrunching

  // coincidencer socket mode
  const char * coincidencer_host;   // host for coincidence reporting
  int          coincidencer_port;   // port for coincidence reporting
 
  // channel zapping
  unsigned int num_channel_zaps;
  hd_range_t * channel_zaps;

  // TESTING
  //hd_size first_beam;
  hd_size beam_count;
  int gpu_id;
  time_t utc_start;             // UTC time of first sample
  hd_size spectra_per_second;   
  const char * output_dir;
};
