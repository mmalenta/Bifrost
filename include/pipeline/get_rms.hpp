/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <pipeline/pipeline_types.hpp>
#include <pipeline/error.hpp>

#include <boost/shared_ptr.hpp>

struct GetRMSPlan_impl;

struct GetRMSPlan {
	GetRMSPlan();
	hd_float exec(hd_float* d_data, hd_size count);
private:
	boost::shared_ptr<GetRMSPlan_impl> m_impl;
};

// Convenience functions for one-off calls
hd_float get_rms(hd_float* d_data, hd_size count);
hd_error normalise(hd_float* d_data, hd_size count);
