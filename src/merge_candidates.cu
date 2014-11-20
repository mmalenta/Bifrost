/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <pipeline/merge_candidates.hpp>
#include <utils/exceptions.hpp>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

typedef thrust::tuple<hd_float,
                      hd_size,hd_size,hd_size,
                      hd_size,hd_size,hd_size> candidate_tuple;
struct merge_candidates_functor : public thrust::binary_function<candidate_tuple,
                                                                 candidate_tuple,
                                                                 candidate_tuple> {
	inline __host__ __device__
	candidate_tuple operator()(const candidate_tuple& c1,
	                           const candidate_tuple& c2) const {
		hd_float snr1 = thrust::get<0>(c1);
		hd_size  ind1 = thrust::get<1>(c1);
		hd_size  begin1 = thrust::get<2>(c1);
		hd_size  end1 = thrust::get<3>(c1);
		hd_size  filter_ind1 = thrust::get<4>(c1);
		hd_size  dm_ind1 = thrust::get<5>(c1);
		hd_size  members1 = thrust::get<6>(c1);
		
		hd_float snr2 = thrust::get<0>(c2);
		hd_size  ind2 = thrust::get<1>(c2);
		hd_size  begin2 = thrust::get<2>(c2);
		hd_size  end2 = thrust::get<3>(c2);
		hd_size  filter_ind2 = thrust::get<4>(c2);
		hd_size  dm_ind2 = thrust::get<5>(c2);
		hd_size  members2 = thrust::get<6>(c2);
		
		if( snr1 >= snr2 ) {
			return thrust::make_tuple(snr1,
			                          ind1,
			                          //(begin1+begin2)/2,
			                          //(end1+end2)/2,
			                          // TODO: I think this is what gtools does
			                          //min((int)begin1, (int)begin2),
			                          //max((int)end1, (int)end2),
			                          // TODO: But this may be better
			                          begin1,
			                          end1,
			                          filter_ind1,
			                          dm_ind1,
			                          members1+members2);
		}
		else {
			return thrust::make_tuple(snr2,
			                          ind2,
			                          //(begin1+begin2)/2,
			                          //(end1+end2)/2,
			                          //min((int)begin1, (int)begin2),
			                          //max((int)end1, (int)end2),
			                          begin2,
			                          end2,
			                          filter_ind2,
			                          dm_ind2,
			                          members1+members2);
		}
	}
};

hd_error merge_candidates(hd_size            count,
                          hd_size*           d_labels,
                          ConstRawCandidates d_cands,
                          RawCandidates      d_groups)
{
	typedef thrust::device_ptr<hd_float> float_iterator;
	typedef thrust::device_ptr<hd_size>  size_iterator;
	typedef thrust::device_ptr<const hd_float> const_float_iterator;
	typedef thrust::device_ptr<const hd_size>  const_size_iterator;
	
	size_iterator  labels_begin(d_labels);
	
	const_float_iterator cand_peaks_begin(d_cands.peaks);
	const_size_iterator  cand_inds_begin(d_cands.inds);
	const_size_iterator  cand_begins_begin(d_cands.begins);
	const_size_iterator  cand_ends_begin(d_cands.ends);
	const_size_iterator  cand_filter_inds_begin(d_cands.filter_inds);
	const_size_iterator  cand_dm_inds_begin(d_cands.dm_inds);
	const_size_iterator  cand_members_begin(d_cands.members);
	
	float_iterator group_peaks_begin(d_groups.peaks);
	size_iterator  group_inds_begin(d_groups.inds);
	size_iterator  group_begins_begin(d_groups.begins);
	size_iterator  group_ends_begin(d_groups.ends);
	size_iterator  group_filter_inds_begin(d_groups.filter_inds);
	size_iterator  group_dm_inds_begin(d_groups.dm_inds);
	size_iterator  group_members_begin(d_groups.members);
	
	// Sort by labels and remember permutation
	thrust::device_vector<hd_size> d_permutation(count);
	thrust::sequence(d_permutation.begin(), d_permutation.end());
	thrust::sort_by_key(labels_begin, labels_begin + count,
	                    d_permutation.begin());
	
	// Merge giants into groups according to the label
	using thrust::reduce_by_key;
	using thrust::make_zip_iterator;
	using thrust::make_permutation_iterator;
	reduce_by_key(labels_begin, labels_begin + count,
	              make_permutation_iterator(
	                  make_zip_iterator(thrust::make_tuple(cand_peaks_begin,
	                                                       cand_inds_begin,
	                                                       cand_begins_begin,
	                                                       cand_ends_begin,
	                                                       cand_filter_inds_begin,
	                                                       cand_dm_inds_begin,
	                                                       cand_members_begin)),
	              d_permutation.begin()),
	              thrust::make_discard_iterator(), // keys output
	              make_zip_iterator(thrust::make_tuple(group_peaks_begin,
	                                                   group_inds_begin,
	                                                   group_begins_begin,
	                                                   group_ends_begin,
	                                                   group_filter_inds_begin,
	                                                   group_dm_inds_begin,
	                                                   group_members_begin)),
	              thrust::equal_to<hd_size>(),
	              merge_candidates_functor());
	
	return HD_NO_ERROR;
}
