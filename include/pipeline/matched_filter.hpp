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

template<typename T>
struct MatchedFilterPlan_impl;

template<typename T>
struct MatchedFilterPlan {
  typedef T value_type;

  MatchedFilterPlan();
  hd_error prep(const T* d_in, hd_size count, hd_size max_width);
  // Note: This writes (count + 1 - max_width) values to d_out
  //         with a relative starting offset of max_width/2
  // Note: This does not apply any normalisation to the output
  hd_error exec(T* d_out, hd_size width, hd_size tscrunch=1);

private:
  boost::shared_ptr<MatchedFilterPlan_impl<T> > m_impl;
};
