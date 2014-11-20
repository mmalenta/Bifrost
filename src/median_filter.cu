/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <pipeline/median_filter.hpp>
#include <kernels/kernels.h>

#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>

/*
  Note: The implementations of median3-5 here can be derived from
          'sorting networks'.
 */

inline __host__ __device__
float median3(float a, float b, float c) {
	return a < b ? b < c ? b
	                      : a < c ? c : a
	             : a < c ? a
	                     : b < c ? c : b;
}
inline __host__ __device__
float median4(float a, float b, float c, float d) {
	return a < c ? b < d ? a < b ? c < d ? 0.5f*(b+c) : 0.5f*(b+d)
	                             : c < d ? 0.5f*(a+c) : 0.5f*(a+d)
	                     : a < d ? c < b ? 0.5f*(d+c) : 0.5f*(b+d)
	                             : c < b ? 0.5f*(a+c) : 0.5f*(a+b)
	             : b < d ? c < b ? a < d ? 0.5f*(b+a) : 0.5f*(b+d)
	                             : a < d ? 0.5f*(a+c) : 0.5f*(c+d)
	                     : c < d ? a < b ? 0.5f*(d+a) : 0.5f*(b+d)
	                             : a < b ? 0.5f*(a+c) : 0.5f*(c+b);
}
inline __host__ __device__
float median5(float a, float b, float c, float d, float e) {
	// Note: This wicked code is by 'DRBlaise' and was found here:
	//         http://stackoverflow.com/a/2117018
	return b < a ? d < c ? b < d ? a < e ? a < d ? e < d ? e : d
                                                 : c < a ? c : a
                                         : e < d ? a < d ? a : d
                                                 : c < e ? c : e
                                 : c < e ? b < c ? a < c ? a : c
                                                 : e < b ? e : b
                                         : b < e ? a < e ? a : e
                                                 : c < b ? c : b
                         : b < c ? a < e ? a < c ? e < c ? e : c
                                                 : d < a ? d : a
                                         : e < c ? a < c ? a : c
                                                 : d < e ? d : e
                                 : d < e ? b < d ? a < d ? a : d
                                                 : e < b ? e : b
                                         : b < e ? a < e ? a : e
                                                 : d < b ? d : b
                 : d < c ? a < d ? b < e ? b < d ? e < d ? e : d
                                                 : c < b ? c : b
                                         : e < d ? b < d ? b : d
                                                 : c < e ? c : e
                                 : c < e ? a < c ? b < c ? b : c
                                                 : e < a ? e : a
                                         : a < e ? b < e ? b : e
                                                 : c < a ? c : a
                         : a < c ? b < e ? b < c ? e < c ? e : c
                                                 : d < b ? d : b
                                         : e < c ? b < c ? b : c
                                                 : d < e ? d : e
                                 : d < e ? a < d ? b < d ? b : d
                                                 : e < a ? e : a
                                         : a < e ? b < e ? b : e
                                                 : d < a ? d : a;
}

struct median_filter3_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	unsigned int    count;
	median_filter3_kernel(const hd_float* in_,
	                      unsigned int count_)
		: in(in_), count(count_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		// Note: We shrink the window near boundaries
		if( i > 0 && i < count-1 ) {
			return median3(in[i-1], in[i], in[i+1]);
		}
		else if( i == 0 ) {
			return 0.5f*(in[i]+in[i+1]);
		}
		else { //if( i == count-1 ) {
			return 0.5f*(in[i]+in[i-1]);
		}
	}
};

struct median_filter5_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	unsigned int    count;
	median_filter5_kernel(const hd_float* in_,
	                      unsigned int count_)
		: in(in_), count(count_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		// Note: We shrink the window near boundaries
		if( i > 1 && i < count-2 ) {
			return median5(in[i-2], in[i-1], in[i], in[i+1], in[i+2]);
		}
		else if( i == 0 ) {
			return median3(in[i], in[i+1], in[i+2]);
		}
		else if( i == 1 ) {
			return median4(in[i-1], in[i], in[i+1], in[i+2]);
		}
		else if( i == count-1 ) {
			return median3(in[i], in[i-1], in[i-2]);
		}
		else { //if ( i == count-2 ) {
			return median4(in[i+1], in[i], in[i-1], in[i-2]);
		}
	}
};

struct median_scrunch3_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	median_scrunch3_kernel(const hd_float* in_)
		: in(in_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_float a = in[3*i+0];
		hd_float b = in[3*i+1];
		hd_float c = in[3*i+2];
		return median3(a, b, c);
	}
};

struct median_scrunch5_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	median_scrunch5_kernel(const hd_float* in_)
		: in(in_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_float a = in[5*i+0];
		hd_float b = in[5*i+1];
		hd_float c = in[5*i+2];
		hd_float d = in[5*i+3];
		hd_float e = in[5*i+4];
		return median5(a, b, c, d, e);
	}
};

struct median_scrunch3_array_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	const hd_size   size;
	median_scrunch3_array_kernel(const hd_float* in_, hd_size size_)
		: in(in_), size(size_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(3*array+0)*size + j];
		hd_float b = in[(3*array+1)*size + j];
		hd_float c = in[(3*array+2)*size + j];
		return median3(a, b, c);
	}
};

struct median_scrunch5_array_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	const hd_size   size;
	median_scrunch5_array_kernel(const hd_float* in_, hd_size size_)
		: in(in_), size(size_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(5*array+0)*size + j];
		hd_float b = in[(5*array+1)*size + j];
		hd_float c = in[(5*array+2)*size + j];
		hd_float d = in[(5*array+3)*size + j];
		hd_float e = in[(5*array+4)*size + j];
		return median5(a, b, c, d, e);
	}
};

hd_error median_filter3(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out)
{
	thrust::device_ptr<hd_float> d_out_begin(d_out);
	using thrust::make_counting_iterator;
	thrust::transform(make_counting_iterator<unsigned int>(0),
	                  make_counting_iterator<unsigned int>(count),
	                  d_out_begin,
	                  median_filter3_kernel(d_in, count));
	return HD_NO_ERROR;
}

hd_error median_filter5(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out)
{
	thrust::device_ptr<hd_float> d_out_begin(d_out);
	using thrust::make_counting_iterator;
	thrust::transform(make_counting_iterator<unsigned int>(0),
	                  make_counting_iterator<unsigned int>(count),
	                  d_out_begin,
	                  median_filter5_kernel(d_in, count));
	return HD_NO_ERROR;
}

hd_error median_scrunch3(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out)
{
	thrust::device_ptr<const hd_float> d_in_begin(d_in);
	thrust::device_ptr<hd_float>       d_out_begin(d_out);
	if( count == 1 ) {
		*d_out_begin = d_in_begin[0];
	}
	else if( count == 2 ) {
		*d_out_begin = 0.5f*(d_in_begin[0] + d_in_begin[1]);
	}
	else {
		// Note: Truncating here is necessary
		hd_size out_count = count / 3;
		using thrust::make_counting_iterator;
		thrust::transform(make_counting_iterator<unsigned int>(0),
		                  make_counting_iterator<unsigned int>(out_count),
		                  d_out_begin,
		                  median_scrunch3_kernel(d_in));
	}
	return HD_NO_ERROR;
}

/* hd_error median_scrunch5(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out)
{
	thrust::device_ptr<const hd_float> d_in_begin(d_in);
	thrust::device_ptr<hd_float>       d_out_begin(d_out);
	
	if( count == 1 ) {
		*d_out_begin = d_in_begin[0];
	}
	else if( count == 2 ) {
		*d_out_begin = 0.5f*(d_in_begin[0] + d_in_begin[1]);
	}
	else if( count == 3 ) {
		*d_out_begin = median3(d_in_begin[0],
		                       d_in_begin[1],
		                       d_in_begin[2]);
	}
	else if( count == 4 ) {
		*d_out_begin = median4(d_in_begin[0],
		                       d_in_begin[1],
		                       d_in_begin[2],
		                       d_in_begin[3]);
	}
	else {
		// Note: Truncating here is necessary
		hd_size out_count = count / 5;
		using thrust::make_counting_iterator;
		thrust::transform(make_counting_iterator<unsigned int>(0),
		                  make_counting_iterator<unsigned int>(out_count),
		                  d_out_begin,
		                  median_scrunch5_kernel(d_in));
	}
	return HD_NO_ERROR;
} */

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 3
hd_error median_scrunch3_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out)
{
	thrust::device_ptr<hd_float> d_out_begin(d_out);
	// Note: Truncating here is necessary
	hd_size out_count = count / 3;
	hd_size total     = array_size * out_count;
	using thrust::make_counting_iterator;
	thrust::transform(make_counting_iterator<unsigned int>(0),
	                  make_counting_iterator<unsigned int>(total),
	                  d_out_begin,
	                  median_scrunch3_array_kernel(d_in, array_size));
	return HD_NO_ERROR;
}

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 5
hd_error median_scrunch5_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out)
{
	thrust::device_ptr<hd_float> d_out_begin(d_out);
	// Note: Truncating here is necessary
	hd_size out_count = count / 5;
	hd_size total     = array_size * out_count;
	using thrust::make_counting_iterator;
	thrust::transform(make_counting_iterator<unsigned int>(0),
	                  make_counting_iterator<unsigned int>(total),
	                  d_out_begin,
	                  median_scrunch5_array_kernel(d_in, array_size));
	return HD_NO_ERROR;
}

template<typename T>
struct mean2_functor : public thrust::binary_function<T,T,T> {
	inline __host__ __device__
	T operator()(T a, T b) const { return (T)0.5 * (a+b); }
};

struct mean_scrunch2_array_kernel
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	const hd_size   size;
	mean_scrunch2_array_kernel(const hd_float* in_, hd_size size_)
		: in(in_), size(size_) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(2*array+0)*size + j];
		hd_float b = in[(2*array+1)*size + j];
		return (hd_float)0.5 * (a+b);
	}
};

// Note: This can operate 'in-place'
hd_error mean_filter2(const hd_float* d_in,
                      hd_size         count,
                      hd_float*       d_out)
{
	thrust::device_ptr<const hd_float> d_in_begin(d_in);
	thrust::device_ptr<hd_float>       d_out_begin(d_out);
	thrust::adjacent_difference(d_in_begin, d_in_begin+count,
	                            d_out_begin,
	                            mean2_functor<hd_float>());
	return HD_NO_ERROR;
}

hd_error mean_scrunch2_array(const hd_float* d_in,
                             hd_size         array_size,
                             hd_size         count,
                             hd_float*       d_out)
{
	thrust::device_ptr<hd_float> d_out_begin(d_out);
	// Note: Truncating here is necessary
	hd_size out_count = count / 2;
	hd_size total     = array_size * out_count;
	using thrust::make_counting_iterator;
	thrust::transform(make_counting_iterator<unsigned int>(0),
	                  make_counting_iterator<unsigned int>(total),
	                  d_out_begin,
	                  mean_scrunch2_array_kernel(d_in, array_size));
	return HD_NO_ERROR;
}

struct linear_stretch_functor
	: public thrust::unary_function<hd_float,hd_float> {
	const hd_float* in;
	hd_float        step;
	linear_stretch_functor(const hd_float* in_,
	                       hd_size in_count, hd_size out_count)
		: in(in_), step(hd_float(in_count-1)/(out_count-1)) {}
	inline __host__ __device__
	hd_float operator()(unsigned int i) const {
		hd_float     x = i * step;
		unsigned int j = x;
		return in[j] + ((x-j > 1e-5) ? (x-j)*(in[j+1]-in[j]) : 0.f);
	}
};

/* hd_error linear_stretch(const hd_float* d_in,
                        hd_size         in_count,
                        hd_float*       d_out,
                        hd_size         out_count)
{
	using thrust::make_counting_iterator;
	thrust::device_ptr<hd_float> d_out_begin(d_out);
	
	thrust::transform(make_counting_iterator<unsigned int>(0),
	                  make_counting_iterator<unsigned int>(out_count),
	                  d_out_begin,
	                  linear_stretch_functor(d_in, in_count, out_count));
	return HD_NO_ERROR;
} */
