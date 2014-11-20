/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#include <pipeline/error.hpp>
#include <pipeline/pipeline_types.hpp>
#include <utils/exceptions.hpp>

hd_error median_filter3(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out);

hd_error median_filter5(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out);

hd_error median_scrunch3(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out);
// in peasoup kernels
/*hd_error median_scrunch5_heimdall(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out);*/

// Note: This can operate 'in-place'
hd_error mean_filter2(const hd_float* d_in,
                      hd_size         count,
                      hd_float*       d_out);
// in peasoup kernels
/*hd_error linear_stretch_hemdaill(const hd_float* d_in,
                        hd_size         in_count,
                        hd_float*       d_out,
                        hd_size         out_count);*/

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 3
hd_error median_scrunch3_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out);

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 5
extern hd_error median_scrunch5_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out);

// Mean-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 2
hd_error mean_scrunch2_array(const hd_float* d_in,
                             hd_size         array_size,
                             hd_size         count,
                             hd_float*       d_out);

#endif
