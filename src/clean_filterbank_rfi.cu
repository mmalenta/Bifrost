/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <pipeline/clean_filterbank_rfi.hpp>
#include <pipeline/remove_baseline.hpp>
#include <pipeline/get_rms.hpp>
#include <pipeline/measure_bandpass.hpp>
#include <pipeline/matched_filter.hpp>
#include <utils/exceptions.hpp>

#include <iostream>
#include <vector>
#include <dedisp.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

// TESTING ONLY
//#include "hd/write_time_series.h"

// A simple hashing function taken from Thrust's Monte Carlo example
inline __host__ __device__
unsigned int hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

template<typename T>
struct abs_less_than : public thrust::unary_function<T,bool> {
  T thresh;
  abs_less_than(T thresh_) : thresh(thresh_) {}
  inline __host__ __device__
  bool operator()(T x) const {
    return fabs(x) < thresh;
  }
};

template<typename WordType>
struct zap_fb_rfi_functor : public thrust::unary_function<WordType,WordType> {
  // Note: Increasing this trades performance for accuracy
  enum { MAX_RESAMPLE_ATTEMPTS = 10 };
  const int*      mask;
  const WordType* in;
  unsigned int    stride;
  unsigned int    nbits;
  unsigned int    nsamps;
  unsigned int    max_resample_dist;
  WordType        bitmask;
  zap_fb_rfi_functor(const int* mask_, const WordType* in_,
                     unsigned int stride_, unsigned int nbits_,
                     unsigned int nsamps_, unsigned int max_resample_dist_)
    : mask(mask_), in(in_),
      stride(stride_), nbits(nbits_), bitmask((1<<nbits)-1),
      nsamps(nsamps_), max_resample_dist(max_resample_dist_) {}
  inline __host__ __device__
  WordType operator()(unsigned int i) const {
    // Lift the 1D index into 2D filterbank coords
    // Note: c is the word, not the channel
    unsigned int t = i / stride;
    unsigned int c = i % stride;
    WordType result;
    if( mask[t] ) {
      unsigned int seed = hash(i);
      // Create a random number engine for this thread
      // Note: This technique is succeptible to correlation between values
      //         A better, but slower, method is to use rng.discard( )
      // TODO: Consider passing a global seed (e.g., derived from the current
      //          time) in here to ensure good randomness.
      thrust::default_random_engine rng(seed);
      result = 0;
      // Iterate over channels in the word
      for( int k=0; k<sizeof(WordType)*8; k+=nbits ) {
        unsigned int min_t = t > max_resample_dist ?
          t - max_resample_dist : 0;
        unsigned int max_t = t < nsamps-1 - max_resample_dist ?
          t + max_resample_dist : nsamps-1;
        thrust::uniform_int_distribution<unsigned int> dist(min_t, max_t);
        unsigned int new_t = dist(rng);
        // Avoid replacing with another bad sample
        // Note: We must limit the number of attempts here for speed
        int attempts = 0;
        while( mask[new_t] && ++attempts < MAX_RESAMPLE_ATTEMPTS+1 ) {
          new_t = dist(rng);
        }

        WordType val = (in[new_t*stride + c] >> k) & bitmask;
        result |= val << k;
      }
    }
    else {
      // Return the input value unchanged
      result = in[i/*t*stride + c*/];
    }
    return result;
  }
};
template<typename WordType>
struct zap_narrow_rfi_functor : public thrust::unary_function<WordType,WordType> {
  // Note: Increasing this trades performance for accuracy
  enum { MAX_RESAMPLE_ATTEMPTS = 10 };
  WordType*       data;
  const float*    baseline;
  float           thresh;
  unsigned int    stride;
  unsigned int    nbits;
  unsigned int    nchans;
  unsigned int    max_resample_dist;
  WordType        bitmask;
  unsigned int    chans_per_word;
  zap_narrow_rfi_functor(WordType* data_, const float* baseline_,
                         float thresh_,
                         unsigned int stride_, unsigned int nbits_,
                         unsigned int nchans_, unsigned int max_resample_dist_)
    : data(data_), baseline(baseline_), thresh(thresh_),
      stride(stride_), nbits(nbits_), bitmask((1<<nbits)-1),
      nchans(nchans_), max_resample_dist(max_resample_dist_),
      chans_per_word(sizeof(WordType)*8/nbits) {}
  
  inline __host__ __device__
  WordType sample(unsigned int t, unsigned int c) const {
    unsigned int w = c / chans_per_word;
    unsigned int k = c % chans_per_word;
    return (data[t*stride + w] >> (k*nbits)) & bitmask;
  }
  
  inline __host__ __device__
  void operator()(unsigned int i) const {
    // Lift the 1D index into 2D filterbank coords
    unsigned int t = i / stride;
    unsigned int w = i % stride;
    WordType word = data[i];
    
    unsigned int seed = hash(i);
    // Create a random number engine for this thread
    // Note: This technique is succeptible to correlation between values
    //         A better, but slower, method is to use rng.discard( )
    // TODO: Consider passing a global seed (e.g., derived from the current
    //          time) in here to ensure good randomness.
    thrust::default_random_engine rng(seed);
    
    bool any_bad = false;
    // Iterate over channels in the word
    //for( int k=0; k<sizeof(WordType)*8; k+=nbits ) {
    for( int k=0; k<chans_per_word; ++k ) {
      unsigned int c = w + k;
      WordType val = (word >> (k*nbits)) & bitmask;
      if( fabs(val - baseline[c]) > thresh ) {
        any_bad = true;

        unsigned int min_c = c > max_resample_dist ?
          c - max_resample_dist : 0;
        unsigned int max_c = c < nchans-1 - max_resample_dist ?
          c + max_resample_dist : nchans-1;
        
        thrust::uniform_int_distribution<unsigned int> distn(min_c, max_c);
        unsigned int new_c = distn(rng);
        
        // Avoid replacing with another bad sample
        // Note: We must limit the number of attempts here for speed
        int attempts = 0;
        WordType new_val = sample(t, new_c);
        while( fabs(new_val - baseline[new_c]) > thresh &&
               ++attempts < MAX_RESAMPLE_ATTEMPTS+1 ) {
          new_c = distn(rng);
          new_val = sample(t, new_c);
        }
        // Replace the relevant bits
        word &= ~(bitmask << (k*nbits));
        word |= new_val << (k*nbits);
      }
    }
    if( any_bad ) {
      data[i] = word;
    }
  }
};

// Zaps the whole band for each masked time sample, replacing values with
//   others sampled randomly from nearby.
hd_error zap_filterbank_rfi(const int* h_mask, const hd_byte* h_in,
                            hd_size nsamps, hd_size nbits, hd_size nchans,
                            hd_size max_resample_dist,
                            hd_byte* h_out)
{
  unsigned int stride_bytes = nchans * nbits / 8;
  
  // Note: This type is used to optimise memory accesses
  //         It also sets the upper limit on nbits
  typedef unsigned int WordType;
  // TODO: Does this break things when nbits > 8 ?
  //typedef hd_byte WordType;
  // Note: This is the stride in words
  // TODO: This assumes the byte stride is a multiple of the word size,
  //         which may not be true.
  unsigned int stride = stride_bytes / sizeof(WordType);
  
  // TODO: Tidy this up. Could possibly pass device arrays rather than host.
  
  // Copy filterbank data to the device
  thrust::device_vector<WordType> d_in((WordType*)h_in,
                                       (WordType*)h_in + nsamps*stride);
  thrust::device_vector<WordType> d_out(nsamps*stride);
  thrust::device_vector<int>      d_mask(h_mask, h_mask+nsamps);
  WordType* d_in_ptr   = thrust::raw_pointer_cast(&d_in[0]);
  int*      d_mask_ptr = thrust::raw_pointer_cast(&d_mask[0]);
  thrust::transform(thrust::counting_iterator<unsigned int>(0),
                    thrust::counting_iterator<unsigned int>(nsamps*stride),
                    d_out.begin(),
                    zap_fb_rfi_functor<WordType>(d_mask_ptr, d_in_ptr,
                                                 stride, nbits,
                                                 nsamps, max_resample_dist));
  // Copy back to the host
  thrust::copy(d_out.begin(), d_out.end(),
               (WordType*)h_out);
  
  return HD_NO_ERROR;
}

template<typename T>
struct is_rfi : public thrust::unary_function<T, bool> {
  T thresh;
  is_rfi(T thresh_) : thresh(thresh_) {}
  inline __host__ __device__
  bool operator()(T x) const {
    return fabs(x) > thresh;
  }
};

template<typename T>
struct rfi_mask_functor : public thrust::binary_function<T,int,bool> {
  T thresh;
  rfi_mask_functor(T thresh_) : thresh(thresh_) {}
  inline __host__ __device__
  bool operator()(T x, int mask) const {
    return (fabs(x) > thresh) || mask;
  }
};

hd_error clean_filterbank_rfi(dedisp_plan    main_plan,
                              const hd_byte* h_in,
                              hd_size        nsamps,
                              hd_size        nbits,
                              hd_byte*       h_out,
                              int*           h_killmask,
                              hd_float       dm,
                              hd_float       dt,
                              hd_float       baseline_length,
                              hd_float       rfi_tol,
                              hd_size        rfi_min_beams,
                              hd_size        boxcar_max)
{
  using thrust::counting_iterator;

  hd_error error;

  typedef hd_float out_type;
  std::vector<out_type>           h_raw_series;
  thrust::device_vector<hd_float> d_series;
  //thrust::host_vector<hd_float>   h_series;
  thrust::device_vector<hd_float> d_filtered;
  //thrust::host_vector<hd_float>   h_beams_series;
  //thrust::device_vector<hd_float> d_beams_series;
  thrust::device_vector<int>      d_filtered_rfi_mask;
  thrust::device_vector<int>      d_rfi_mask;
  thrust::host_vector<int>        h_rfi_mask;

  hd_size nchans = dedisp_get_channel_count(main_plan);

  // TODO: Any way to avoid having to use this?
  thrust::host_vector<hd_byte> h_in_copy;

  typedef unsigned int WordType;
  hd_size stride = nchans * nbits/8 / sizeof(WordType);

  // TODO: Any way to avoid having to use this?
  thrust::device_vector<WordType> d_in((WordType*)h_in,
                                       (WordType*)h_in + nsamps*stride);
  WordType* d_in_ptr = thrust::raw_pointer_cast(&d_in[0]);

  thrust::device_vector<hd_float> d_bandpass(nchans);
  hd_float* d_bandpass_ptr = thrust::raw_pointer_cast(&d_bandpass[0]);

  // Narrow-band RFI is not an issue when nbits is small
  // Note: Small nbits can actually cause this excision code to fail
  if( nbits > 4 ) {
    // Narrow-band RFI excision
    // ------------------------
    // TODO: Any motivation for this?
    //       Make it a parameter?

	std::cout << "Using this lovely for loop..." << std::endl;
	hd_size max_chan_resample_dist = nchans / 60;

    // We loop over gulps of nsamps_smooth samples so that each one
    //   gets its own bandpass measurement.
    // TODO: Should this be halved? (Note: adds 25% to total cleaning time)
    hd_size nsamps_smooth = hd_size(baseline_length / (1 * dt));
    for( hd_size g=0; g<nsamps; g+=nsamps_smooth ) {
      hd_size nsamps_gulp = std::min(nsamps_smooth, nsamps-g);

      // Measure the bandpass
      hd_float rms = 0;
      measure_bandpass((hd_byte*)(d_in_ptr + g*stride),
                       nsamps_gulp, nchans, nbits,
                       d_bandpass_ptr, &rms);

      zap_narrow_rfi_functor<WordType> zapit(d_in_ptr,
                                             d_bandpass_ptr,
                                             rfi_tol*rms,
                                             stride, nbits, nchans,
                                             max_chan_resample_dist);

      // Zap narrow-band RFI
      counting_iterator<unsigned int> begin(g*stride);
      counting_iterator<unsigned int> end((g+nsamps_gulp)*stride);
      thrust::for_each(begin, end,
                       zapit);
    }

    h_in_copy.resize(nsamps*stride*sizeof(WordType));
    thrust::copy(d_in.begin(), d_in.end(),
                 (WordType*)&h_in_copy[0]);
  }
  else {
	std::cout << "Not using this lovely for loop..." << std::endl;
    h_in_copy.assign(h_in, h_in+nsamps*nchans*nbits/8);
  }
  // ------------------------

  // Broad-band RFI excision
  // First, dedisperse at the given DM
  // ---------------------------------
  dedisp_error derror;
  // Create a new plan for the zero-DM dedispersion
  dedisp_float f0 = dedisp_get_f0(main_plan);
  dedisp_float df = dedisp_get_df(main_plan);
  dedisp_plan plan;
  derror = dedisp_create_plan(&plan, nchans, dt, f0, df);
  if( derror != DEDISP_NO_ERROR ) {
    return throw_dedisp_error(derror);
  }

  derror = dedisp_disable_adaptive_dt(plan);
  if( derror != DEDISP_NO_ERROR ) {
    return throw_dedisp_error(derror);
  }
  derror = dedisp_set_dm_list(plan, &dm, 1);
  if( derror != DEDISP_NO_ERROR ) {
    return throw_dedisp_error(derror);
  }
  hd_size max_delay       = dedisp_get_max_delay(plan);
  hd_size nsamps_computed = nsamps - max_delay;

  h_raw_series.resize(nsamps_computed);

  unsigned flags = DEDISP_USE_DEFAULT;
  const dedisp_byte* in        = (const dedisp_byte*)&h_in_copy[0];
  dedisp_byte*       out       = (dedisp_byte*)&h_raw_series[0];
  hd_size            out_nbits = sizeof(out_type)*8;
  derror = dedisp_execute(plan, nsamps,
                          in, nbits,// in_stride,
                          out, out_nbits,// out_stride,
                          //gulp_dm, dm_gulp_size,
                          flags);
  if( derror != DEDISP_NO_ERROR ) {
    return throw_dedisp_error(derror);
  }
  dedisp_destroy_plan(plan);
  // ---------------------------------

  // Then baseline and normalise the time series
  // -------------------------------------------
  // Copy to the device and convert to floats
  d_series = h_raw_series;
  // Remove the baseline
  hd_size nsamps_smooth = hd_size(baseline_length / (2 * dt));
  hd_float* d_series_ptr = thrust::raw_pointer_cast(&d_series[0]);

  //write_device_time_series(d_series_ptr, nsamps_computed,
  //                         dt, "dm0_dedispersed.tim");

  RemoveBaselinePlan baseline_remover;
  error = baseline_remover.exec(d_series_ptr, nsamps_computed, nsamps_smooth);
  if( error != HD_NO_ERROR ) {
    return throw_error(error);
  }

  //write_device_time_series(d_series_ptr, nsamps_computed,
  //                         dt, "dm0_baselined.tim");

  // Normalise
  error = normalise(d_series_ptr, nsamps_computed);
  if( error != HD_NO_ERROR ) {
    return throw_error(error);
  }

  //write_device_time_series(d_series_ptr, nsamps_computed,
  //                         dt, "dm0_normalised.tim");
  // -------------------------------------------

  // Do a simple sigma cut to identify RFI
  // -------------------------------------
  d_rfi_mask.resize(nsamps_computed, 0);

  d_filtered_rfi_mask.resize(nsamps_computed, 0);
  int* d_filtered_rfi_mask_ptr =
    thrust::raw_pointer_cast(&d_filtered_rfi_mask[0]);

  // Create an RFI mask for this filter
  thrust::transform(d_series.begin(), d_series.end(),
                    d_rfi_mask.begin(),
                    is_rfi<hd_float>(rfi_tol));
  
  // Note: The filtered output is shorter by boxcar_max samps
  //         and offset by boxcar_max/2 samps.
  d_filtered.resize(nsamps_computed + 1 - boxcar_max);
  hd_float* d_filtered_ptr = thrust::raw_pointer_cast(&d_filtered[0]);
  MatchedFilterPlan<hd_float> filter_plan;
  filter_plan.prep(d_series_ptr, nsamps_computed, boxcar_max);
  
  for( hd_size filter_width=1; filter_width<=boxcar_max;
       filter_width*=2 ) {
    
    // Apply the matched filter
    // Note: The filtered output is shorter by boxcar_max samps
    //         and offset by (boxcar_max-1)/2+1 samps.
    filter_plan.exec(d_filtered_ptr, filter_width);
    
    // Normalise the filtered time series (RMS ~ sqrt(time))
    thrust::constant_iterator<hd_float> 
      norm_val_iter(1.0 / sqrt((hd_float)filter_width));
    thrust::transform(d_filtered.begin(),
                      d_filtered.end(),
                      norm_val_iter,
                      d_filtered.begin(),
                      thrust::multiplies<hd_float>());
    
    //hd_size filter_offset = (boxcar_max-1)/2+1;
    hd_size filter_offset = boxcar_max / 2;
    
    // Create an RFI mask for this filter
    thrust::transform(d_filtered.begin(), d_filtered.end(),
                      d_filtered_rfi_mask.begin() + filter_offset,
                      is_rfi<hd_float>(rfi_tol));
    
    // Filter the RFI mask
    // Note: This ensures we zap all samples contributing to the peak
    MatchedFilterPlan<int> mask_filter_plan;
    mask_filter_plan.prep(d_filtered_rfi_mask_ptr, nsamps_computed,
                          boxcar_max);
    mask_filter_plan.exec(d_filtered_rfi_mask_ptr + filter_offset,
                          filter_width);
    
    // Merge the filtered mask with the global mask
    thrust::transform(d_rfi_mask.begin(), d_rfi_mask.end(),
                      d_filtered_rfi_mask.begin(),
                      d_rfi_mask.begin(),
                      thrust::logical_or<int>());
  }
  h_rfi_mask = d_rfi_mask;
  // -------------------------------------
  
  // Finally, apply the mask to zap RFI in the filterbank
  error = zap_filterbank_rfi(&h_rfi_mask[0],
                             &h_in_copy[0],
                             nsamps_computed,
                             nbits,
                             nchans,
                             // TODO: This is somewhat arbitrary
                             nsamps_smooth/4,
                             &h_out[0]);
  if( error != HD_NO_ERROR ) {
    return error;
  }
  
  return HD_NO_ERROR;
}

hd_error apply_manual_killmasks (dedisp_plan    main_plan,
                                 int*           h_killmask,
                                 unsigned int num_channel_zaps,
                                 hd_range_t * channel_zaps)
{
  hd_size nchans = dedisp_get_channel_count(main_plan);
  for (unsigned i=0; i<num_channel_zaps; i++)
  {
    for (unsigned j=channel_zaps[i].start; j<channel_zaps[i].end; j++)
    {
      if (j < nchans)
        h_killmask[j] = 0;
    }
  }
  return HD_NO_ERROR;
}
