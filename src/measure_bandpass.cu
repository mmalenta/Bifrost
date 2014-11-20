/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <kernels/kernels.h>
#include <pipeline/measure_bandpass.hpp>
#include <pipeline/median_filter.hpp>
#include <pipeline/get_rms.hpp>

// TESTING ONLY
// #include "hd/write_time_series.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>

template<typename WordType>
struct unpack_functor : public thrust::unary_function<unsigned int, float> {
	const WordType* in;
	unsigned int    nbits;
	unsigned int    chans_per_word;
	WordType        bitmask;
	unpack_functor(const WordType* in_, unsigned int nbits_)
		: in(in_), nbits(nbits_),
		  chans_per_word(sizeof(WordType)*8/nbits), bitmask((1<<nbits)-1) {}
	
	inline __host__ __device__
	float operator()(unsigned int c) const {
		unsigned int w = c / chans_per_word;
		unsigned int k = c % chans_per_word;
		return (float)((in[w] >> (k*nbits)) & bitmask);
	}
};

template<typename T>
struct abs_val : public thrust::unary_function<T,T> {
	inline __host__ __device__
	T operator()(T x) const { return fabs(x); }
};
hd_error measure_bandpass(const hd_byte* d_filterbank,
                          hd_size        nsamps,
                          hd_size        nchans,
                          hd_size        nbits,
                          hd_float*      d_bandpass,
                          hd_float*      rms)
{
	using thrust::make_counting_iterator;
	
	typedef unsigned int WordType;
	hd_size stride = nchans * nbits/8 / sizeof(WordType);
	
	//thrust::device_vector<hd_float> d_spectrum(nchans);
	//hd_float* d_spectrum_ptr = thrust::raw_pointer_cast(&d_spectrum[0]);
	thrust::device_ptr<hd_float> d_bandpass_begin(d_bandpass);
	
	// First we find the median of a selection of sample spectra
	// TODO: Can/should make this a parameter?
	// TODO: Does this give a good balance of performance vs. accuracy?
	// Note: Changing this requires changing the code below.
	hd_size spectrum_count = 5*5*5 *5*5;
	thrust::device_vector<hd_float> d_sample_spectra1(spectrum_count*nchans);
	thrust::device_vector<hd_float> d_sample_spectra2(spectrum_count/5*nchans);
	thrust::device_vector<hd_float> d_sample_spectra3(spectrum_count/5/5*nchans);
	thrust::device_vector<hd_float> d_sample_spectra4(spectrum_count/5/5/5*nchans);
	thrust::device_vector<hd_float> d_sample_spectra5(spectrum_count/5/5/5/5*nchans);
	
	hd_float* d_sample_spectra1_ptr =
		thrust::raw_pointer_cast(&d_sample_spectra1[0]);
	hd_float* d_sample_spectra2_ptr =
		thrust::raw_pointer_cast(&d_sample_spectra2[0]);
	hd_float* d_sample_spectra3_ptr =
		thrust::raw_pointer_cast(&d_sample_spectra3[0]);
	hd_float* d_sample_spectra4_ptr =
		thrust::raw_pointer_cast(&d_sample_spectra4[0]);
	hd_float* d_sample_spectra5_ptr =
		thrust::raw_pointer_cast(&d_sample_spectra5[0]);
	
	// TODO: Make this more random?
	hd_size seed = 123456;
	thrust::default_random_engine rng(seed);
	thrust::uniform_int_distribution<unsigned int> distribution(0, nsamps-1);
	// Extract spectrum_count sample spectra from the filterbank
	for( hd_size i=0; i<spectrum_count; ++i ) {
		//hd_size t = i * spectrum_stride; // Regular spacing
		hd_size t = distribution(rng); // Uniform random sampling
		WordType* d_in = (WordType*)&d_filterbank[t*stride];
		thrust::transform(make_counting_iterator<unsigned int>(0),
		                  make_counting_iterator<unsigned int>(nchans),
		                  d_sample_spectra1.begin() + i*nchans,
		                  unpack_functor<WordType>(d_in, nbits));
	}
	
	// Compute the 'remedian' (recursive median) of the sample spectra
	// Note: We do this instead of a proper median for performance and simplicity
	median_scrunch5_array(d_sample_spectra1_ptr, nchans,
	                      spectrum_count,
	                      d_sample_spectra2_ptr);
	median_scrunch5_array(d_sample_spectra2_ptr, nchans,
	                      spectrum_count / 5,
	                      d_sample_spectra3_ptr);
	median_scrunch5_array(d_sample_spectra3_ptr, nchans,
	                      spectrum_count / 5 / 5,
	                      d_sample_spectra4_ptr);
	median_scrunch5_array(d_sample_spectra4_ptr, nchans,
	                      spectrum_count / 5 / 5 / 5,
	                      d_sample_spectra5_ptr);
	median_scrunch5_array(d_sample_spectra5_ptr, nchans,
	                      spectrum_count / 5 / 5 / 5 / 5,
	                      d_bandpass);
	
	//write_device_time_series(d_bandpass, nchans, 1.f, "median_spectrum.tim");
	
	// Now we smooth the spectrum to produce an estimate of the bandpass
	thrust::device_vector<hd_float> d_scrunched_spectrum(nchans/5);
	hd_float* d_scrunched_spectrum_ptr =
		thrust::raw_pointer_cast(&d_scrunched_spectrum[0]);
	// TODO: This algorithm was derived empirically. It may not be suitable
	//         if applied to a different observing setup.
	median_scrunch5(d_bandpass, nchans,
	                d_scrunched_spectrum_ptr);
	median_filter5(d_scrunched_spectrum_ptr, nchans / 5,
	               d_bandpass);
	mean_filter2(d_bandpass, nchans / 5,
	             d_scrunched_spectrum_ptr);
	linear_stretch(d_scrunched_spectrum_ptr, nchans / 5,
	               d_bandpass,
	               // Note: We must use the truncate-rounded length
	               nchans / 5 * 5);
	
	// Extrapolate to make up the truncated samples
	// Note: This is very inefficient, but shouldn't affect performance
	for( hd_size i=nchans/5*5; i<nchans; ++i ) {
		d_bandpass_begin[i] =
			2 * d_bandpass_begin[i-1] - d_bandpass_begin[i-2];
	}
	// The bandpass estimate is now in d_bandpass
	
	//write_device_time_series(d_bandpass, nchans, 1.f, "bandpass.tim");
	
	// Now we estimate the RMS in the bandpass
	// ---------------------------------------
	//std::vector<hd_float> sample_rms1(spectrum_count);
	//std::vector<hd_float> sample_rms2(spectrum_count);
	// TODO: These are on the device only because I couldn't be bothered
	//         making host versions of the median_scrunch functions.
	thrust::device_vector<hd_float> d_sample_rms1(spectrum_count);
	thrust::device_vector<hd_float> d_sample_rms2(spectrum_count);
	hd_float* d_sample_rms1_ptr =
		thrust::raw_pointer_cast(&d_sample_rms1[0]);
	hd_float* d_sample_rms2_ptr =
		thrust::raw_pointer_cast(&d_sample_rms2[0]);
	
	for( hd_size i=0; i<spectrum_count; ++i ) {
		// Subtract the bandpass from the spectrum
		thrust::transform(d_sample_spectra1.begin() + i*nchans,
		                  d_sample_spectra1.begin() + (i+1)*nchans,
		                  d_bandpass_begin,
		                  d_sample_spectra1.begin() + i*nchans,
		                  thrust::minus<hd_float>());
		// Take the absolute value
		thrust::transform(d_sample_spectra1.begin() + i*nchans,
		                  d_sample_spectra1.begin() + (i+1)*nchans,
		                  d_sample_spectra1.begin() + i*nchans,
		                  abs_val<hd_float>());
		
		//d_sample_rms1[i] = get_rms(d_sample_spectra1_ptr + i*nchans, nchans);
	}
	
	thrust::device_vector<hd_float> d_mad(nchans);
	hd_float* d_mad_ptr = thrust::raw_pointer_cast(&d_mad[0]);
	
	// Compute the 'remedian' (recursive median) of the sample spectra
	// Note: We do this instead of a proper median for performance and simplicity
	median_scrunch5_array(d_sample_spectra1_ptr, nchans,
	                      spectrum_count,
	                      d_sample_spectra2_ptr);
	median_scrunch5_array(d_sample_spectra2_ptr, nchans,
	                      spectrum_count / 5,
	                      d_sample_spectra3_ptr);
	median_scrunch5_array(d_sample_spectra3_ptr, nchans,
	                      spectrum_count / 5 / 5,
	                      d_sample_spectra4_ptr);
	median_scrunch5_array(d_sample_spectra4_ptr, nchans,
	                      spectrum_count / 5 / 5 / 5,
	                      d_sample_spectra5_ptr);
	median_scrunch5_array(d_sample_spectra5_ptr, nchans,
	                      spectrum_count / 5 / 5 / 5 / 5,
	                      d_mad_ptr);
	
	// Convert median absolute deviation to standard deviation
	using namespace thrust::placeholders;
	thrust::transform(d_mad.begin(), d_mad.end(),
	                  d_mad.begin(),
	                  _1 * 1.4826f);
	
	//write_device_time_series(d_mad_ptr, nchans, 1.f, "mad.tim");
	/*
	// Smooth the band RMS
	median_scrunch5(d_mad_ptr, nchans,
	                d_scrunched_spectrum_ptr);
	median_filter5(d_scrunched_spectrum_ptr, nchans / 5,
	               d_mad_ptr);
	mean_filter2(d_mad_ptr, nchans / 5,
	             d_scrunched_spectrum_ptr);
	linear_stretch(d_scrunched_spectrum_ptr, nchans / 5,
	               d_mad_ptr,
	               // Note: We must use the truncate-rounded length
	               nchans / 5 * 5);
	
	// Extrapolate to make up the truncated samples
	// Note: This is very inefficient, but shouldn't affect performance
	for( hd_size i=nchans/5*5; i<nchans; ++i ) {
		d_mad[i] = 2 * d_mad[i-1] - d_mad[i-2];
	}
	*/
	// TODO: Do we need to apply narrow-band filtering to (all of the)
	//         time-scrunched versions of the filterbank too?
	//         This would allow us to catch narrow, extended RFI.
	//       What about scrunching in frequency a bit too?
	//         Probably a bad idea, as that's what the broad-band mitigation
	//           is for, and we want as much distinction as possible.
	
	//write_device_time_series(d_mad_ptr, nchans, 1.f, "smooth_mad.tim");
	
	// Find the median RMS across the band
	std::vector<hd_float> h_mad(nchans);
	thrust::copy(d_mad.begin(), d_mad.end(), h_mad.begin());
	std::nth_element(h_mad.begin(), h_mad.begin()+h_mad.size()/2, h_mad.end());
	*rms = h_mad[h_mad.size()/2];
	
	/*
	// And finally use the remedian to estimate the global RMS
	median_scrunch5(d_sample_rms1_ptr, spectrum_count,
	                d_sample_rms2_ptr);
	median_scrunch5(d_sample_rms2_ptr, spectrum_count / 5,
	                d_sample_rms1_ptr);
	median_scrunch5(d_sample_rms1_ptr, spectrum_count / 5 / 5,
	d_sample_rms2_ptr);
	*rms = d_sample_rms2[0];
	*/
	// ---------------------------------------
	
	return HD_NO_ERROR;
}

/*
// TODO: The below code is work in progress.

hd_error measure_band_avg(const hd_byte* d_filterbank,
                          hd_size        nsamps,
                          hd_size        nchans,
                          hd_size        nbits,
                          hd_float*      d_band_avg)
{
	using thrust::make_counting_iterator;
	
	typedef unsigned int WordType;
	hd_size stride = nchans * nbits/8 / sizeof(WordType);
	
	thrust::device_vector<hd_float> d_bandpass(nchans);
	hd_float* d_bandpass_ptr = raw_pointer_cast(&d_bandpass[0]);
	hd_float rms;
	measure_bandpass(d_filterbank, nsamps, nchans, nbits,
	                 d_bandpass_ptr, &rms);
	
	// TODO: Check that this gives good results, good performance and not
	//         too much memory use.
	hd_size spectrum_count = 1<<11;
	thrust::device_vector<hd_float> d_sample_spectra(spectrum_count*nchans);
	thrust::device_vector<hd_float> d_scrunched_spectra(spectrum_count/2*nchans);
	using thrust::raw_pointer_cast;
	hd_float* d_sample_spectra_ptr =
		raw_pointer_cast(&d_sample_spectra[0]);
	hd_float* d_scrunched_spectra_ptr =
		raw_pointer_cast(&d_scrunched_spectra[0]);
	// TODO: Make this more random?
	hd_size seed = 123456;
	thrust::default_random_engine rng(seed);
	thrust::uniform_int_distribution<unsigned int> distribution(0, nsamps-1);
	// Extract spectrum_count sample spectra from the filterbank
	for( hd_size i=0; i<spectrum_count; ++i ) {
		//hd_size t = i * spectrum_stride; // Regular spacing
		hd_size t = distribution(rng); // Uniform random sampling
		WordType* d_in = (WordType*)&d_filterbank[t*stride];
		// Extract the spectrum
		thrust::transform(make_counting_iterator<unsigned int>(0),
		                  make_counting_iterator<unsigned int>(nchans),
		                  d_sample_spectra.begin() + i*nchans,
		                  unpack_functor<WordType>(d_in, nbits));
		// Subtract the bandpass
		thrust::transform(d_sample_spectra.begin() + i*nchans,
		                  d_sample_spectra.begin() + (i+1)*nchans,
		                  d_bandpass.begin(),
		                  d_sample_spectra.begin() + i*nchans,
		                  thrust::minus<hd_float>());
	}
	
	for( hd_size count=spectrum_count; count>1; count/=2 ) {
		mean_scrunch2_array(d_sample_spectra_ptr, nchans,
		                    count,
		                    d_scrunched_spectra_ptr);
		
		std::swap(d_sample_spectra_ptr, d_scrunched_spectra_ptr);
	}
	thrust::device_ptr<hd_float> d_band_sum_begin(d_sample_spectra_ptr);
	thrust::device_ptr<hd_float> d_band_sum_end = d_band_sum_begin + nchans;
	thrust::device_ptr<hd_float> d_band_avg_begin(d_band_avg);
	using namespace thrust::placeholders;
	thrust::transform(d_band_sum_begin, d_band_sum_end,
	                  thrust::make_constant_iterator((float)sqrt(spectrum_count)),
	                  d_band_avg_begin,
	                  thrust::multiplies<hd_float>());
	                  //_1 * sqrt((float)spectrum_count));
	
	return HD_NO_ERROR;
}
*/
/*
// Measures each channel's outlier fraction, i.e., the fraction of samples
//   in the channel that exceed thresh*rms.
hd_error measure_band_outliers(const hd_byte* d_filterbank,
                               hd_size        nsamps,
                               hd_size        nchans,
                               hd_size        nbits,
                               hd_float       thresh,
                               hd_float*      d_band_outlier_fracs)
{
	using thrust::make_counting_iterator;
	
	typedef unsigned int WordType;
	hd_size stride = nchans * nbits/8 / sizeof(WordType);
	
	thrust::device_vector<hd_float> d_bandpass(nchans);
	hd_float* d_bandpass_ptr = raw_pointer_cast(&d_bandpass[0]);
	hd_float rms;
	measure_bandpass(d_filterbank, nsamps, nchans, nbits,
	                 d_bandpass_ptr, &rms);
	
	// TODO: Check that this gives good results, good performance and not
	//         too much memory use.
	hd_size spectrum_count = 1<<10;
	thrust::device_vector<hd_float> d_sample_spectra(spectrum_count*nchans);
	thrust::device_vector<hd_float> d_scrunched_spectra(spectrum_count/2*nchans);
	using thrust::raw_pointer_cast;
	hd_float* d_sample_spectra_ptr =
		raw_pointer_cast(&d_sample_spectra[0]);
	hd_float* d_scrunched_spectra_ptr =
		raw_pointer_cast(&d_scrunched_spectra[0]);
	// TODO: Make this more random?
	hd_size seed = 123456;
	thrust::default_random_engine rng(seed);
	thrust::uniform_int_distribution<unsigned int> distribution(0, nsamps-1);
	// Extract spectrum_count sample spectra from the filterbank
	for( hd_size i=0; i<spectrum_count; ++i ) {
		//hd_size t = i * spectrum_stride; // Regular spacing
		hd_size t = distribution(rng); // Uniform random sampling
		WordType* d_in = (WordType*)&d_filterbank[t*stride];
		// Extract spectrum
		thrust::transform(make_counting_iterator<unsigned int>(0),
		                  make_counting_iterator<unsigned int>(nchans),
		                  d_sample_spectra.begin() + i*nchans,
		                  unpack_functor<WordType>(d_in, nbits));
		// Transform to a mask of outliers
		thrust::transform(d_sample_spectra.begin() + i*nchans,
		                  d_sample_spectra.begin() + (i+1)*nchans,
		                  d_bandpass.begin(),
		                  d_sample_spectra.begin() + i*nchans,
		                  abs_diff_exceeds<hd_float>(thresh*rms));
		// TODO: transform to mask of fabs(x-xb) > thresh, where xb is the bandpass
		//       Then leave the below scrunching as is, remove the final
		//         normalisation transform, and check if the result exceeds
		//         the desired time infection percentage.
	}
	
	for( hd_size count=spectrum_count; count>1; count/=2 ) {
		mean_scrunch2_array(d_sample_spectra_ptr, nchans,
		                    count,
		                    d_scrunched_spectra_ptr);
		
		std::swap(d_sample_spectra_ptr, d_scrunched_spectra_ptr);
	}
	thrust::device_ptr<hd_float> d_band_sum_begin(d_sample_spectra_ptr);
	thrust::device_ptr<hd_float> d_band_sum_end = d_band_sum_begin + nchans;
	thrust::device_ptr<hd_float> d_band_outlier_fracs_begin(d_band_outlier_fracs);
	using namespace thrust::placeholders;
	
	// Copy to output
	thrust::transform(d_band_sum_begin, d_band_sum_end,
	                  d_band_outlier_fracs_begin,
	                  thrust::identity<hd_float>());
	
	return HD_NO_ERROR;
}
*/
