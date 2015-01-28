/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <kernels/kernels.h>
#include <pipeline/get_rms.hpp>
#include <pipeline/median_filter.hpp>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

template<typename T>
struct absolute_val : public thrust::unary_function<T,T> {
	inline __host__ __device__
	T operator()(T x) const { return abs(x); }
};

class GetRMSPlan_impl {
	thrust::device_vector<hd_float> buf1;
	thrust::device_vector<hd_float> buf2;
public:
	hd_float exec(hd_float* d_data, hd_size count) {
		thrust::device_ptr<hd_float> d_data_begin(d_data);
		
		// This algorithm works by taking the absolute values of the data
		//   and then repeatedly scrunching them using median-of-5 in order
		//   to approximate the median absolute deviation. The RMS is then
		//   just 1.4862 times this - relationship between RMS and MAD
		//   for normal distribution
		
		buf1.resize(count);
		buf2.resize(count/5);
		hd_float* buf1_ptr = thrust::raw_pointer_cast(&buf1[0]);
		hd_float* buf2_ptr = thrust::raw_pointer_cast(&buf2[0]);
		
		thrust::transform(d_data_begin, d_data_begin+count,
		                  buf1.begin(),
		                  absolute_val<hd_float>());
		
		for( hd_size size=count; size>1; size/=5 ) {
			median_scrunch5(buf1_ptr, size, buf2_ptr);
			std::swap(buf1_ptr, buf2_ptr);
		}
		// Note: Result is now at buf1_ptr
		thrust::device_ptr<hd_float> buf1_begin(buf1_ptr);
		hd_float med_abs_dev = buf1_begin[0];
		hd_float rms = med_abs_dev * 1.4862;
		
		return rms;
	}
};

// Public interface (wrapper for implementation)
GetRMSPlan::GetRMSPlan()
	: m_impl(new GetRMSPlan_impl) {}
hd_float GetRMSPlan::exec(hd_float* d_data, hd_size count) {
	return m_impl->exec(d_data, count);
}

// Convenience functions for one-off calls
hd_float get_rms(hd_float* d_data, hd_size count) {
	return GetRMSPlan().exec(d_data, count);
}
hd_error normalise(hd_float* d_data, hd_size count)
{
	thrust::device_ptr<hd_float> d_data_begin(d_data);
	thrust::device_ptr<hd_float> d_data_end(d_data + count);
	
	hd_float rms = get_rms(d_data, count);
	thrust::transform(d_data_begin, d_data_end,
	                  thrust::make_constant_iterator(hd_float(1.0)/rms),
	                  d_data_begin,
	                  thrust::multiplies<hd_float>());
	
	return HD_NO_ERROR;
}
