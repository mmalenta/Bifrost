/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *   Modified by Mateusz Malenta to include in Bifrost
 ***************************************************************************/



#pragma once

#include <cstddef>

typedef unsigned char         hd_byte;
typedef size_t                hd_size;
//typedef unsigned int          hd_size;
typedef float                 hd_float;
typedef struct hd_pipeline_t* hd_pipeline;

// Fundamental candidate quantities only
struct RawCandidates {
	hd_float* peaks;
	hd_size*  inds;
	hd_size*  begins;
	hd_size*  ends;
	hd_size*  filter_inds;
	hd_size*  dm_inds;
	hd_size*  members;
};
struct ConstRawCandidates {
	const hd_float* peaks;
	const hd_size*  inds;
	const hd_size*  begins;
	const hd_size*  ends;
	const hd_size*  filter_inds;
	const hd_size*  dm_inds;
	const hd_size*  members;
};
// Full candidate info including derived quantities
struct Candidates : public RawCandidates {
	hd_float* dms;
	hd_size*  flags;
	hd_size*  beam_counts;
	hd_size*  beam_masks;
};

typedef struct hd_range {
  hd_size start;
  hd_size end;
} hd_range_t;


