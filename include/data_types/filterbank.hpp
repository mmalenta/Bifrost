/*
  Copyright 2014 Ewan Barr
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
  http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
/*
  filterbank.hpp

  By Ewan Barr (2013)
  ewan.d.barr@gmail.com

  This file contians classes and methods for the reading, storage
  and manipulation of filterbank format data. Filterbank format 
  can be any time-frequency data block. Time must be the slowest 
  changing dimension.
*/

#pragma once
#include <algorithm>
//#include <cstdint>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <vector>
#include "data_types/header.hpp"
#include "utils/exceptions.hpp"
#include <pipeline/pipeline_types.hpp>
/*!
  \brief Base class for handling filterbank data.

  All time and frequency resolved data types should inherit
  from this class. Class presents virtual set and get methods 
  for various requrired meta data. The filterbank data itself
  is stred in the *data pointer as unsigend chars.
*/
class Filterbank {
protected:
  //Filterbank metadata
  unsigned char*  data; /*!< Pointer to filterbank data.*/ 
  unsigned int nsamps; /*!< Number of time samples. */ 
  unsigned int nchans; /*!< Number of frequecy channels. */ 
  unsigned char nbits; /*!< Bits per time sample. */ 
  float fch1; /*!< Frequency of top channel (MHz) */ 
  float foff; /*!< Channel bandwidth (MHz) */ 
  float tsamp; /*!< Sampling time (seconds) */ 
  
  /*!
    \brief Instantiate a new Filterbank object with metadata.
    
    Instantiate a new Filterbank object from an existing data
    pointer and metadata.
    
    \param data_ptr A pointer to a memory location containing filterbank data.
    \param nsamps The number of time samples in the data.
    \param nchans The number of frequency channels in that data.
    \param nbins The size of a single data point in bits.
    \param fch1 The centre frequency of the first data channel.
    \param foff The bandwidth of a frequency channel.
    \param tsamp The sampling time of the data.
  */
  Filterbank(unsigned char* data_ptr, unsigned int nsamps,
	     unsigned int nchans, unsigned char nbits,
	     float fch1, float foff, float tsamp)

    :data(data_ptr),nsamps(nsamps),nchans(nchans),
     nbits(nbits),fch1(fch1),foff(foff),tsamp(tsamp){}
  
  /*!
    \brief Instantiate a new default Filterbank object.
    
    Create a new Filterbank object with the data pointer and 
    all metadata set to zero.
  */
  Filterbank(void)
    :data(0),nsamps(0),nchans(0),
     nbits(0),fch1(0.0),foff(0.0),tsamp(0.0){}

public:
  
  /*!
    \brief Get the currently set sampling time.
    
    \return The currently set sampling time.
  */
  virtual float get_tsamp(void){return tsamp;}
  
  /*!
    \brief Set the sampling time.
    
    \param tsamp The sampling time of the data (in seconds).
  */
  virtual void set_tsamp(float tsamp){this->tsamp = tsamp;}

  /*!
    \brief Get the currently set channel bandwidth.
    
    \return The channel bandwidth (in MHz).
  */
  virtual float get_foff(void){return foff;}
    
  /*!
    \brief Set the channel bandwidth.

    \param foff The channel bandwidth (in MHz).
  */
  virtual void set_foff(float foff){this->foff = foff;}

  /*!
  \brief Get the frequency of the top channel.

  \return The frequency of channel 0 (in MHz)
  */
  virtual float get_fch1(void){return fch1;}
  
  /*!
    \brief Set the frequency of the top channel.

    \param fch1 The frequency of channel 0 (in MHz).
  */
  virtual void set_fch1(float fch1){this->fch1 = fch1;}
  
  /*!
    \brief Get the number of frequency channels.

    \return The number of frequency channels.
  */
  virtual float get_nchans(void){return nchans;}

  /*!
    \brief Set the number of frequency channels.

    \param nchans The number of frequency channels in the data.
  */
  virtual void set_nchans(unsigned int nchans){this->nchans = nchans;}

  /*!
    \brief Get the number of time samples in the data.

    \return The number of time samples.
  */
  virtual unsigned int get_nsamps(void){return nsamps;}
  
  /*!
    \brief Set the number of time samples in data.

    \param nsamps The number of time samples.
  */
  virtual void set_nsamps(unsigned int nsamps){this->nsamps = nsamps;}

  /*!
    \brief Get the number of bits per sample.

    \return The number of bits per sample.
  */
  virtual float get_nbits(void){return nbits;}
  
  /*!
    \brief Set the number of bits per sample.

    \param nbits The number of bits per sample.
  */
  virtual void set_nbits(unsigned char nbits){this->nbits = nbits;}

  /*!
    \brief Get the pointer to the filterbank data.
    
    \return The pointer to the filterbank data.
  */
  virtual unsigned char * get_data(void){return this->data;}
  
  virtual size_t get_data_range(size_t nsamps, hd_byte *vector_data)
  {
//	std::cout << "Testing:" << std::endl << "data[0]" << this->data[0] << std::endl << "data[1]" << this->data[1] << std::endl;
	size_t nchan_bytes = (nchans * nbits) / (8 * sizeof(char));
	std::copy(data, data + (nsamps * nchan_bytes), vector_data);
	size_t bytes_read = nsamps * nchan_bytes;	// for now
	return bytes_read / nchan_bytes;
  }

  /*!
    \brief Set the filterbank data pointer.

    \param data A pointer to a block of filterbank data.
  */
  virtual void set_data(unsigned char *data){this->data = data;}
  
  /*!
  \brief Get the centre frequency of the data block.

  \return The centre frequency of the filterbank data.
  */
  virtual float get_cfreq(void)
  {
    if (foff < 0)
      return fch1+foff*nchans/2;
    else
      return fch1-foff*nchans/2;
  }
};


/*!
  \brief A class for handling Sigproc format filterbanks.
  
  A subclass of the Filterbank class for handling filterbank
  in Sigproc style/format from file. Filterbank memory buffer
  is allocated in constructor and deallocated in destructor.
*/
class SigprocFilterbank: public Filterbank {
public:
  /*!
    \brief Create a new SigprocFilterbank object from a file.
    
    Constructor opens a filterbank file reads the header and then
    reads all of the data from the filterbank file into CPU RAM.
    Metadata is set from the filterbank header values.

    \param filename Path to a valid sigproc filterbank file.
  */
  SigprocFilterbank(std::string filename)
  {
    std::ifstream infile;
    SigprocHeader hdr;
    infile.open(filename.c_str(),std::ifstream::in | std::ifstream::binary);
    ErrorChecker::check_file_error(infile, filename);
    // Read the header
    read_header(infile,hdr);
    size_t input_size = (size_t) hdr.nsamples*hdr.nbits*hdr.nchans/8;

    infile.seekg(hdr.size, std::ios::beg);
    // Read the data

    unsigned char * data_temp = new unsigned char [input_size];

    std::cout << "Reading the file\n";
    infile.read(reinterpret_cast<char*>(data_temp), input_size);
    // Set the metadata
    this->nchans = hdr.nchans;
    this->nbits = hdr.nbits;
    this->fch1 = hdr.fch1;
    this->foff  = hdr.foff;

    std::cout << "The frequency of top channel [MHz]: " << hdr.fch1 << std::endl;
    std::cout << "The channel bandwidht [MHz]: " << hdr.foff << std::endl;
    std::cout << "The number of channels: " << hdr.nchans << std::endl;

    //averaging the time samples

    std::cout << "Averaging time samples\n";

    double new_tsamp = (double)(hdr.tsamp * 2.0);

    unsigned int new_nsamples = (hdr.nsamples / 2); // new number of time samples per channel

    size_t new_input_size = (size_t) new_nsamples * hdr.nbits * hdr.nchans / 8;

    unsigned char *data_new = new unsigned char [new_input_size];

    unsigned int nchans = hdr.nchans;
    unsigned int nsamples = hdr.nsamples;		

    size_t total_nsamples = (size_t) nchans * nsamples;

    unsigned int data_point_1, data_point_2, data_point_3, data_point_4, data_point_5, data_point_6, data_point_7, data_point_8;

    size_t saved = 0;

    for ( size_t current_sample_block = 0; current_sample_block < total_nsamples; current_sample_block+= size_t(nchans * 2))
    { 

	// the following code will only work if the number of channels can be divided by 8
	// need to include a check and introduce unrollinf by 2 if the number of channels cannot be divided by 8
	// no problem for GHRSS though, which makes use of 1024 channels and will make use of 2048 in the future

	for (size_t current_channel = 0; current_channel < nchans; current_channel+=8)
	{
                data_point_1 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel] + data_temp[(size_t) current_sample_block + current_channel + nchans]);
                data_point_2 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel + 1] + data_temp[(size_t) current_sample_block + current_channel + 1 + nchans]);
                data_point_3 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel + 2] + data_temp[(size_t) current_sample_block + current_channel + 2 + nchans]);
                data_point_4 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel + 3] + data_temp[(size_t) current_sample_block + current_channel + 3 + nchans]);
                data_point_5 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel + 4] + data_temp[(size_t) current_sample_block + current_channel + 4 + nchans]);
                data_point_6 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel + 5] + data_temp[(size_t) current_sample_block + current_channel + 5 + nchans]);
                data_point_7 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel + 6] + data_temp[(size_t) current_sample_block + current_channel + 6 + nchans]);
                data_point_8 = (unsigned int) (data_temp[(size_t) current_sample_block + current_channel + 7] + data_temp[(size_t) current_sample_block + current_channel + 7 + nchans]);

		if (data_point_1 > 255)
			data_point_1 = 255;		

		if (data_point_2 > 255)
			data_point_2 = 255;

		if (data_point_3 > 255)
			data_point_3 = 255;
		
		if (data_point_4 > 255)
			data_point_4 = 255;
	
		if (data_point_5 > 255)
			data_point_5 = 255;

		if (data_point_6 > 255)
			data_point_6 = 255;

		if (data_point_7 > 255)
			data_point_7 = 255;

		if (data_point_8 > 255)
			data_point_8 = 255;

		data_new[(size_t) saved + current_channel] = (unsigned char) data_point_1;
		data_new[(size_t) saved + current_channel + 1] = (unsigned char) data_point_2;
		data_new[(size_t) saved + current_channel + 2] = (unsigned char) data_point_3;
		data_new[(size_t) saved + current_channel + 3] = (unsigned char) data_point_4;
		data_new[(size_t) saved + current_channel + 4] = (unsigned char) data_point_5;
		data_new[(size_t) saved + current_channel + 5] = (unsigned char) data_point_6;
		data_new[(size_t) saved + current_channel + 6] = (unsigned char) data_point_7;
		data_new[(size_t) saved + current_channel + 7] = (unsigned char) data_point_8;


	}

        saved += nchans;

    }

   delete [] data_temp;		// cleaning

   this->nsamps = new_nsamples;
   this->tsamp  = new_tsamp;
   this->data   = data_new;   

   std::cout << (int)this->data[0] << std::endl;
   std::cout << this->tsamp << std::endl;

   std::cout << "Finished averaging time samples!\n";

   }

  
  /*!
    \brief Deconstruct a SigprocFilterbank object.
    
    The deconstructor cleans up memory allocated when
    reading data from file. 
  */
  ~SigprocFilterbank()
  {
    delete [] this->data;
  }
};
