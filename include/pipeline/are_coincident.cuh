/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <pipeline/pipeline_types.hpp>

inline __host__ __device__
bool ranges_overlap(hd_size bi, hd_size ei,
                    hd_size bj, hd_size ej,
                    hd_size tol) {
	return bi <= ej+tol && bj <= ei+tol;
}

inline __host__ __device__
bool are_coincident(hd_size samp_i, hd_size samp_j,
                    hd_size begin_i, hd_size begin_j,
                    hd_size end_i, hd_size end_j,
                    hd_size filter_i, hd_size filter_j,
                    hd_size dm_i, hd_size dm_j,
                    hd_size time_tol, hd_size filter_tol, hd_size dm_tol) {
	
	// TODO: Should time_tol be adjusted for the filtering level(s)?
	//         Sarah's giantsearch doesn't seem to use any tol at all.
	
	// TODO: TESTING tolerance proportional to filter width
	//time_tol *= ((1<<filter_i) + (1<<filter_j)) / 2;
	time_tol *= max((int)(1<<filter_i), (int)(1<<filter_j));
	
	// TODO: Avoid the (int) casts?
	//return ranges_overlap(begin_i, end_i, begin_j, end_j, time_tol) &&
	//	abs((int)filter_j - (int)filter_i) <= filter_tol &&
	//	abs((int)dm_j - (int)dm_i ) <= dm_tol;
	// New version avoiding use of 'begin' and 'end'
	return abs((int)samp_j-(int)samp_i) <= time_tol &&
		abs((int)filter_j - (int)filter_i) <= filter_tol &&
		abs((int)dm_j - (int)dm_i ) <= dm_tol;
}
