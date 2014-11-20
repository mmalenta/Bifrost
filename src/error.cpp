/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// TODO: For debugging only
#include <iostream>
using std::cerr;
using std::endl;

#include <pipeline/error.hpp>
#include <utils/exceptions.hpp>

const char* hd_get_error_string(hd_error error)
{
	switch( error ) {
		case HD_NO_ERROR:
			return "No error";
		case HD_MEM_ALLOC_FAILED:
			return "Memory allocation failed";
		case HD_MEM_COPY_FAILED:
			return "Memory copy failed";
		case HD_INVALID_DEVICE_INDEX:
			return "Invalid device index";
		case HD_DEVICE_ALREADY_SET:
			return "Device is already set and cannot be changed";
		//case HD_NCHANS_EXCEEDS_LIMIT:
		//	return "No. channels exceeds internal limit";
		case HD_INVALID_PIPELINE:
			return "Invalid pipeline";
		case HD_INVALID_POINTER:
			return "Invalid pointer";
		case HD_INVALID_STRIDE:
			return "Invalid stride";
		/*
		case HD_NO_DM_LIST_SET:
			return "No DM list has been set";
		case HD_TOO_FEW_NSAMPS:
			return "No. samples < maximum delay";
		case HD_INVALID_FLAG_COMBINATION:
			return "Invalid flag combination";
		case HD_UNSUPPORTED_IN_NBITS:
			return "Unsupported in_nbits value";
		case HD_UNSUPPORTED_OUT_NBITS:
			return "Unsupported out_nbits value";
		*/
		case HD_PRIOR_GPU_ERROR:
			return "Prior GPU error.";
		case HD_INTERNAL_GPU_ERROR:
			return "Internal GPU error. Please contact the author(s).";
    case HD_TOO_MANY_EVENTS:
			return "Too many events";
		case HD_UNKNOWN_ERROR:
			return "Unknown error. Please contact the author(s).";
		default:
			return "Invalid error code";
	}
}

hd_error throw_error(hd_error error) {
	return error;
}
hd_error throw_dedisp_error(dedisp_error error) {
	
	cerr << "DEDISP ERROR: " << dedisp_get_error_string(error) << endl;
	
	switch( error ) {
	case DEDISP_MEM_ALLOC_FAILED:
		return HD_MEM_ALLOC_FAILED;
	case DEDISP_MEM_COPY_FAILED:
		return HD_MEM_COPY_FAILED;
	case DEDISP_INVALID_DEVICE_INDEX:
		return HD_INVALID_DEVICE_INDEX;
	case DEDISP_DEVICE_ALREADY_SET:
		return HD_DEVICE_ALREADY_SET;
	//case DEDISP_INVALID_PIPELINE:
	//	return HD_INVALID_PIPELINE;
	case DEDISP_INVALID_POINTER:
		return HD_INVALID_POINTER;
	case DEDISP_INVALID_STRIDE:
		return HD_INVALID_STRIDE;
	case DEDISP_PRIOR_GPU_ERROR:
		return HD_PRIOR_GPU_ERROR;
	case DEDISP_INTERNAL_GPU_ERROR:
		return HD_INTERNAL_GPU_ERROR;
	// TODO: Translate the rest too
	default:
		return HD_UNKNOWN_ERROR;
	}
}
hd_error throw_cuda_error(cudaError_t error) {
	switch( error ) {
	case cudaErrorInvalidDevice:
		return HD_INVALID_DEVICE_INDEX;
	case cudaErrorSetOnActiveProcess:
		return HD_DEVICE_ALREADY_SET;
	case cudaErrorMemoryAllocation:
		return HD_MEM_ALLOC_FAILED;
	// TODO: Translate the rest too
	default:
		return HD_UNKNOWN_ERROR;
	}
}
