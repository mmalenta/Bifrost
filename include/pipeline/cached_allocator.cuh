/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <thrust/system/cuda/vector.h>

#include <map>
#include <stdexcept>

#include <iostream>

/*
  This was copied from Thrust's custom_temporary_allocation example.
 */

// cached_allocator: a simple allocator for caching allocation requests
struct cached_allocator
{
  cached_allocator() {}

  void *allocate(std::ptrdiff_t num_bytes)
  {
	  //std::cout << "cached_allocator::allocate()" << std::endl;
	  
    void *result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if(free_block != free_blocks.end())
    {
	    //std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase(free_block);
    }
    else
    {
      // no allocation of the right size exists
      // create a new one with cuda::malloc
      // throw if cuda::malloc can't satisfy the request
      try
      {
	      //std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

        // allocate memory and convert cuda::pointer to raw pointer
        result = thrust::system::cuda::malloc(num_bytes).get();
      }
      catch(std::runtime_error &e)
      {
        throw;
      }
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(void *ptr)
  {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

  void free_all()
  {
	  //std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

    // deallocate all outstanding blocks in both lists
    for(free_blocks_type::iterator i = free_blocks.begin();
        i != free_blocks.end();
        ++i)
    {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::system::cuda::free(thrust::system::cuda::pointer<void>(i->second));
    }

    for(allocated_blocks_type::iterator i = allocated_blocks.begin();
        i != allocated_blocks.end();
        ++i)
    {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::system::cuda::free(thrust::system::cuda::pointer<void>(i->first));
    }
  }

  typedef std::multimap<std::ptrdiff_t, void*> free_blocks_type;
  typedef std::map<void *, std::ptrdiff_t>     allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;
};

// Global instance of custom allocator
extern cached_allocator g_allocator;

// Tag for custom memory allocator in Thrust calls
struct my_tag : thrust::system::cuda::tag {};

// overload get_temporary_buffer on my_tag
// its job is to forward allocation requests to g_allocator
template<typename T>
  thrust::pair<T*, std::ptrdiff_t>
    get_temporary_buffer(my_tag, std::ptrdiff_t n)
{
	//std::cout << "get_temporary_buffer" << std::endl;
  // ask the allocator for sizeof(T) * n bytes
  T* result = reinterpret_cast<T*>(g_allocator.allocate(sizeof(T) * n));

  // return the pointer and the number of elements allocated
  return thrust::make_pair(result,n);
}
// overload return_temporary_buffer on my_tag
// its job is to forward deallocations to g_allocator
// an overloaded return_temporary_buffer should always accompany
// an overloaded get_temporary_buffer
template<typename Pointer>
  void return_temporary_buffer(my_tag, Pointer p)
{
	//std::cout << "return_temporary_buffer" << std::endl;
  // return the pointer to the allocator
  g_allocator.deallocate(thrust::raw_pointer_cast(p));
}
