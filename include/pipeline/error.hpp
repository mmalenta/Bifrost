/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <dedisp.h>
#include <cuda_runtime_api.h>

typedef int hd_error;

/* enum {
	HD_NO_ERROR = 0,
	HD_MEM_ALLOC_FAILED,
	HD_MEM_COPY_FAILED,
	HD_INVALID_DEVICE_INDEX,
	HD_DEVICE_ALREADY_SET,
	HD_INVALID_PIPELINE,
	HD_INVALID_POINTER,
	HD_INVALID_STRIDE,
	HD_INVALID_NBITS,
	
	HD_PRIOR_GPU_ERROR,
	HD_INTERNAL_GPU_ERROR,
	
	HD_TOO_MANY_EVENTS,
	// ...
	HD_UNKNOWN_ERROR
}; */

const char* hd_get_error_string(hd_error error);

hd_error throw_error(hd_error error);
hd_error throw_dedisp_error(dedisp_error error);
hd_error throw_cuda_error(cudaError_t error);
