/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <iostream>
#include <fstream>
#include <pipeline/matched_filter.hpp>
#include <pipeline/strided_range.hpp>
#include <utils/exceptions.hpp>

#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>

// TODO: Add error checking to the methods in here
template<typename T>
class MatchedFilterPlan_impl {
	thrust::device_vector<T> m_scanned;
	hd_size                  m_max_width;
	
public:
	hd_error prep(const T* d_in, hd_size count, hd_size max_width) {
		m_max_width = max_width;
		
		thrust::device_ptr<const T> d_in_begin(d_in);
		thrust::device_ptr<const T> d_in_end(d_in + count);
		
		// Note: One extra element so that we include the final value
		m_scanned.resize(count + 1);
		thrust::exclusive_scan(d_in_begin, d_in_end + 1,
		                       m_scanned.begin());

//		std::ofstream scan_out("scan_out.dat", std::ofstream::out | std::ofstream::trunc);
//		std::ofstream orig_data("orig_data.dat", std::ofstream::out | std::ofstream::trunc);


/*		for (size_t i = 0; i < count + 1; i++)
		{
			orig_data << *(d_in_begin + i) << std::endl;
			scan_out << m_scanned[i] << std::endl;
		}
		scan_out.close(); */

		return HD_NO_ERROR;
	}

	// Note: This writes div_round_up(count + 1 - max_width, tscrunch) values to d_out
	//         with a relative starting offset of max_width/2
	// Note: This does not apply any normalisation to the output
	hd_error exec(T* d_out, hd_size filter_width, hd_size tscrunch=1) {
		// TODO: Check that prep( ) has been called
		// TODO: Check that width <= m_max_width
		
		thrust::device_ptr<T> d_out_begin(d_out);
		
		hd_size offset    = m_max_width / 2;

		// ahead and behind are the same except for the case when filter_width = 1
		hd_size ahead     = (filter_width-1)/2+1;   // Divide and round up
		hd_size behind    = filter_width / 2;       // Divide and round down
		hd_size out_count = m_scanned.size() - m_max_width;
		

		hd_size stride = tscrunch;

//		std::cout << "filter width = " << filter_width << std::endl;
//		std::cout << "ahead = " << ahead << std::endl;
//		std::cout << "behind = " << behind << std::endl;
//		std::cout << "stride = " << stride << std::endl;

		typedef typename thrust::device_vector<T>::iterator Iterator;

		// Striding through the scanned array has the same effect as tscrunching
		// TODO: Think about this carefully. Does it do exactly what we want?
		strided_range<Iterator> in_range1(m_scanned.begin()+offset + ahead,
		                                  m_scanned.begin()+offset + ahead + out_count,
		                                  stride);
		strided_range<Iterator> in_range2(m_scanned.begin()+offset - behind,
		                                  m_scanned.begin()+offset - behind + out_count,
		                                  stride);


		thrust::transform(in_range1.begin(), in_range1.end(),
		                  in_range2.begin(),
		                  d_out_begin,
		                  thrust::minus<T>());

/*		if (filter_width == 512)
		{
			std::ofstream first_range("first_range.dat", std::ofstream::out | std::ofstream::trunc);
			std::ofstream second_range("second_range.dat", std::ofstream::out | std::ofstream::trunc);
			std::ofstream difference("difference.dat", std::ofstream::out | std::ofstream::trunc);

			for (size_t i = 0; i < out_count; i++)
			{
				first_range << *(in_range1.begin() + i) << std::endl;
				second_range << *(in_range2.begin() + i) << std::endl;
				difference << *(d_out_begin +i) << std::endl;
			}

			first_range.close();
			second_range.close();
			difference.close();
			
			std::cout << "Saved some data...";
			exit(0);
		} */
		return HD_NO_ERROR;
	}
};

// Public interface (wrapper for implementation)
template<typename T>
MatchedFilterPlan<T>::MatchedFilterPlan() : m_impl(new MatchedFilterPlan_impl<T>) {}
template<typename T>
hd_error MatchedFilterPlan<T>::prep(const T* d_in, hd_size count,
                                    hd_size max_width) {
	return m_impl->prep(d_in, count, max_width);
	//return (*this)->prep(d_in, count, max_width);
}
template<typename T>
hd_error MatchedFilterPlan<T>::exec(T* d_out, hd_size filter_width,
                                    hd_size tscrunch) {
	return m_impl->exec(d_out, filter_width, tscrunch);
}

// Explicit template instantiations for types used by other compilation units
template struct MatchedFilterPlan<hd_float>;
template struct MatchedFilterPlan<int>;
