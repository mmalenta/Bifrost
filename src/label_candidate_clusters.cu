/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <utils/exceptions.hpp>
#include <pipeline/label_candidate_clusters.hpp>
#include <pipeline/are_coincident.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
/*
// Lexicographically projects 3D integer coordinates onto a 1D coordinate
// Also applies an offset and performs boundary clamping
template<typename T>
struct project_coords_functor : public thrust::unary_function<void,T> {
	int width;
	int height;
	int depth;
	int stride_x;
	int stride_y;
	int stride_z;
	int dx;
	int dy;
	int dz;
	project_coords_functor() {}
	project_coords_functor(int w, int h, int d, int rad,
	                       int dx_=0, int dy_=0, int dz_=0)
		: width(w), height(h), depth(d),
		  stride_x(1),
		  stride_y(w),//+2*rad),
		  stride_z(h*(w)),//+2*rad)),
		  dx(dx_), dy(dy_), dz(dz_) {}
	
	template<typename CoordTuple>
	inline __host__ __device__
	T operator()(CoordTuple xyz) const {
		//int x = thrust::get<0>(xyz) + dx;
		// HACK TESTING
		int filter_width = 1 << thrust::get<1>(xyz);
		int x = thrust::get<0>(xyz) + dx * filter_width;
		
		x = min(max(x,0),width-1);
		int y = (int)thrust::get<1>(xyz) + dy;
		int z = (int)thrust::get<2>(xyz) + dz;
		// We must be careful around the borders
		if( z < 0 ) {
			x = 0;
		}
		else if( z > depth-1 ) {
			x = width-1;
		}
		else if( y < 0 ) {
			x = 0;
		}
		else if( y > height-1 ) {
			x = width-1;
		}
		y = min(max(y,0),height-1);
		z = min(max(z,0),depth-1);
		return x * stride_x + y * stride_y + z * stride_z;
	}
};

// Returns the minimum over a specified range of elements
template<typename ValueType, typename SizeType>
struct range_min_functor : public thrust::binary_function<void,void,ValueType> {
	const ValueType* data;
	range_min_functor(const ValueType* data_) : data(data_) {}
	
	template<typename BeginEndTuple>
	inline __host__ __device__
	ValueType operator()(ValueType init, BeginEndTuple begin_end) const {
		SizeType begin = thrust::get<0>(begin_end);
		SizeType end   = thrust::get<1>(begin_end);
		ValueType result = init;
		for( SizeType i=begin; i<end; ++i ) {
			// TODO: What's with the stupid min( ) overloads?
			//result = min((long long)result, (long long)data[i]);
			result = (result <= data[i]) ? result : data[i];
		}
		return result;
	}
};
*/
__device__ unsigned int d_counter;

// Finds the root of a chain of equivalent labels
//   E.g., 3->1, 4->3, 8->4, 5->8 => [1,3,4,5,8]->1
// TODO: It would be quite interesting to study the behaviour of this
//         algorithm/implementation in more detail.
template<typename T>
struct trace_equivalency_chain {
	T* new_labels;
	trace_equivalency_chain(T* new_labels_) : new_labels(new_labels_) {}
	inline /*__host__*/ __device__
	void operator()(unsigned int old_label) const {
		T cur_label = old_label;
		while( new_labels[cur_label] != cur_label ) {
			cur_label = new_labels[cur_label];
			//new_labels[old_label] = cur_label;
			// TESTING TODO: See if/how this varies if we write
			//                 new_labels[old_label] each iteration vs.
			//                 only at the end (see commented line below).
			//               It appears to make only 10-20% difference
			atomicAdd(&d_counter, 1);
		}
		new_labels[old_label] = cur_label;
		
		/*
		T j = i;
		while( new_labels[i] != j ) {
			new_labels[i] = new_labels[j];
		}
		
		
		T j = i;
		T new_label = new_labels[i] = new_labels[j];
		while( new_label != j ) {
			j = new_label;
			//new_label = new_labels[j];
			new_label = new_labels[i] = new_labels[j];
		}
		// Note: This written value may subsequently be read by another thread,
		//         which should improve the speed of the algorithm by exploiting
		//         already-computed results.
		new_labels[i] = new_label;
		*/
	}
};

struct cluster_functor {
	hd_size  count;
	const hd_size* d_samp_inds;
	const hd_size* d_begins;
	const hd_size* d_ends;
	const hd_size* d_filters;
	const hd_size* d_dms;
	hd_size* d_labels;
	hd_size  time_tol;
	hd_size  filter_tol;
	hd_size  dm_tol;
	
	cluster_functor(hd_size count_,
	                const hd_size* d_samp_inds_,
	                const hd_size* d_begins_, const hd_size* d_ends_,
	                const hd_size* d_filters_, const hd_size* d_dms_,
	                hd_size* d_labels_,
	                hd_size time_tol_, hd_size filter_tol_, hd_size dm_tol_)
		: count(count_),
		  d_samp_inds(d_samp_inds_),
		  d_begins(d_begins_), d_ends(d_ends_),
		  d_filters(d_filters_), d_dms(d_dms_),
		  d_labels(d_labels_),
		  time_tol(time_tol_), filter_tol(filter_tol_), dm_tol(dm_tol_) {}
	
	inline __host__ __device__
	void operator()(unsigned int i) {
		hd_size samp_i   = d_samp_inds[i];
		hd_size begin_i  = d_begins[i];
		hd_size end_i    = d_ends[i];
		hd_size filter_i = d_filters[i];
		hd_size dm_i     = d_dms[i];
		// TODO: This would be much faster using shared mem like in nbody
		for( unsigned int j=0; j<count; ++j ) {
			if( j == i ) {
				continue;
			}
			hd_size samp_j   = d_samp_inds[j];
			hd_size begin_j  = d_begins[j];
			hd_size end_j    = d_ends[j];
			hd_size filter_j = d_filters[j];
			hd_size dm_j     = d_dms[j];
			if( are_coincident(samp_i, samp_j,
			                   begin_i, begin_j,
			                   end_i, end_j,
			                   filter_i, filter_j,
			                   dm_i, dm_j,
			                   time_tol, filter_tol, dm_tol) ) {
				// Re-label as the minimum of the two
				d_labels[i] = min((int)d_labels[i], (int)d_labels[j]);
			}
		}
	}
};

// Finds components of the given list that are connected in time, filter and DM
// Note: merge_dist is the distance in time up to which components are connected
// Note: Merge distances in filter and DM space are currently fixed at 1
// TODO: Consider re-naming the *_count args to *_max
hd_error label_candidate_clusters(hd_size            count,
                                  ConstRawCandidates d_cands,
                                  hd_size            time_tol,
                                  hd_size            filter_tol,
                                  hd_size            dm_tol,
                                  hd_size*           d_labels,
                                  hd_size*           label_count)
{

	/*
	  def within_range(bi, ei, bj, ej, tol):
	      return bi <= ej+tol and bj <= ei+tol;
	      //return ej - bi >= 0 and ei - bj >= 0;
	      //return ej - bi >= -tol and ei - bj >= -tol;
	  
	  for ci in candidates:
	      for cj in candidates:
	          if ci == cj:
	              continue
	          if( abs(ci.dm_ind - cj.dm_ind) <= dm_ind_tol &&
	              abs(ci.filter_ind - cj.filter_ind) <= filter_ind_tol &&
	              within_range(ci.begin,ci.end,cj.begin,cj.end,time_tol) ):
	              ci.new_label = min(ci.new_label, cj.new_label);
	 */

	using thrust::make_counting_iterator;

	thrust::device_ptr<hd_size> d_labels_begin(d_labels);
	thrust::sequence(d_labels_begin, d_labels_begin+count);

	// This just does a brute-force O(N^2) search for neighbours and
	//   re-labels as the minimum label over neighbours.
	thrust::for_each(make_counting_iterator<unsigned int>(0),
	                 make_counting_iterator<unsigned int>(count),
	                 cluster_functor(count,
	                                 d_cands.inds,
	                                 d_cands.begins,
	                                 d_cands.ends,
	                                 d_cands.filter_inds,
	                                 d_cands.dm_inds,
	                                 d_labels,
	                                 time_tol,
	                                 filter_tol,
	                                 dm_tol));

	/*
	using thrust::make_transform_iterator;
	using thrust::make_zip_iterator;
	using thrust::make_counting_iterator;

	typedef thrust::device_ptr<const hd_size> const_coord_iterator;
	typedef thrust::device_ptr<hd_size>             coord_iterator;

	const_coord_iterator d_begins_begin(d_begins);
	const_coord_iterator d_ends_begin(d_ends);
	//const_coord_iterator d_beams_begin(d_beams);
	const_coord_iterator d_filters_begin(d_filter_inds);
	const_coord_iterator d_dms_begin(d_dm_inds);
	coord_iterator       d_labels_begin(d_labels);
	
	typedef thrust::device_vector<hd_size> coord_vector;
	coord_vector d_new_labels(count);
	coord_vector d_neib_begins(count);
	coord_vector d_neib_ends(count);
	
	thrust::sequence(d_labels_begin, d_labels_begin+count);
	
	typedef thrust::zip_iterator<thrust::tuple<
	                             const_coord_iterator,
	                             const_coord_iterator,
	                             const_coord_iterator> > coords_iterator;
	
	coords_iterator begin_coords(thrust::make_tuple(d_begins_begin,
	                                                d_filters_begin,
	                                                d_dms_begin));
	coords_iterator end_coords(thrust::make_tuple(d_ends_begin,
	                                              d_filters_begin,
	                                              d_dms_begin));
	
	project_coords_functor<hd_size> project_coords(time_count,
	                                               filter_count,
	                                               dm_count,
	                                               merge_dist);
	project_coords_functor<hd_size> project_offset_coords;
	
	hd_size search_count = 14;
	// Note: This list could be expanded to connect components over greater
	//         dists in filter and DM space (currently 1).
	//int     search_offsets[][2] = {{-1,0}, {0,-1}, {-1,-1}};
	int     search_offsets[][2] = {{-1,0}, {1,0},
	                               {-1,-1}, {0,-1}, {1,-1},
	                               {-1,-2}, {0,-2}, {1,-2},
	                               {-1,-3}, {0,-3}, {1,-3},
	                               {-1,-4}, {0,-4}, {1,-4}};
	for( hd_size i=0; i<search_count; ++i ) {
		//std::cout << "Searching around offset "
		//          << search_offsets[i][0] << ", "
		//          << search_offsets[i][1] << std::endl;
		// Find the beginning of each element's neighbours
		project_offset_coords =
			project_coords_functor<hd_size>(time_count,
			                                filter_count,
			                                dm_count,
			                                merge_dist,
			                                -(int)merge_dist+1,
			                                search_offsets[i][0],
			                                search_offsets[i][1]);
		thrust::lower_bound(make_transform_iterator(end_coords,
		                                            project_coords),
		                    make_transform_iterator(end_coords,
		                                            project_coords)+count,
		                    make_transform_iterator(begin_coords,
		                                            project_offset_coords),
		                    make_transform_iterator(begin_coords,
		                                            project_offset_coords)+count,
		                    d_neib_begins.begin());
		//std::cout << "neib_begins:" << std::endl;
		//thrust::copy(d_neib_begins.begin(), d_neib_begins.end(),
		//             std::ostream_iterator<hd_size>(std::cout, "\t"));
		//std::cout << std::endl;
		
		// Find the end of each element's neighbours
		project_offset_coords =
			project_coords_functor<hd_size>(time_count+merge_dist*2,
			                                filter_count,
			                                dm_count,
			                                merge_dist,
			                                +merge_dist,
			                                search_offsets[i][0],
			                                search_offsets[i][1]);
		thrust::upper_bound(make_transform_iterator(begin_coords,
		                                            project_coords),
		                    make_transform_iterator(begin_coords,
		                                            project_coords)+count,
		                    make_transform_iterator(end_coords,
		                                            project_offset_coords),
		                    make_transform_iterator(end_coords,
		                                            project_offset_coords)+count,
		                    d_neib_ends.begin());
		//std::cout << "neib_ends:" << std::endl;
		//thrust::copy(d_neib_ends.begin(), d_neib_ends.end(),
		//             std::ostream_iterator<hd_size>(std::cout, "\t"));
		//std::cout << std::endl;
		
		// Now find the minimum label over each element's neighbours
		thrust::transform(d_labels_begin, d_labels_begin+count,
		                  make_zip_iterator(thrust::make_tuple(d_neib_begins.begin(),
		                                                       d_neib_ends.begin())),
		                  d_new_labels.begin(),
		                  range_min_functor<hd_size,hd_size>(d_labels));
		//std::cout << "new_labels:" << std::endl;
		//thrust::copy(d_new_labels.begin(), d_new_labels.end(),
		//             std::ostream_iterator<hd_size>(std::cout, "\t"));
		//std::cout << std::endl;

		// Copy the latest labels to the
		thrust::copy(d_new_labels.begin(), d_new_labels.end(),
		             d_labels_begin);
	}
	*/
	// Finally, trace equivalency chains to find the final labels
	// Note: This is a parallel version of this algorithm that may not be
	//         as efficient as the sequential version but should win out
	//         in overall speed.


	unsigned int* d_counter_address;
	cudaGetSymbolAddress((void**)&d_counter_address, d_counter);

	thrust::device_ptr<unsigned int> d_counter_ptr(d_counter_address);
	*d_counter_ptr = 0;

	thrust::for_each(make_counting_iterator<unsigned int>(0),
	                 make_counting_iterator<unsigned int>(count),
	                 trace_equivalency_chain<hd_size>(d_labels));


	//std::cout << "Total chain iterations: " << *d_counter_ptr << std::endl;

	// Finally we do a quick count of the number of unique labels
	//   This is efficiently achieved by checking where new labels are
	//     unchanged from their original values (i.e., where d_labels[i] == i)
	thrust::device_vector<int> d_label_roots(count);
	thrust::transform(d_labels_begin, d_labels_begin+count,
	                  make_counting_iterator<hd_size>(0),
	                  d_label_roots.begin(),
	                  thrust::equal_to<hd_size>());
	*label_count = thrust::count_if(d_label_roots.begin(),
	                                d_label_roots.end(),
	                                thrust::identity<hd_size>());

	return HD_NO_ERROR;
}
