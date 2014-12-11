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
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <vector>
#include "data_types/header.hpp"
#include "utils/exceptions.hpp"
#include <pipeline/pipeline_types.hpp>

#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
using std::cout;
using std::endl;

extern int current;
int channel_mod (); // { current++; int modulus = current % 1024; if (modulus == 0) return 1024;
		//		else return modulus;}

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
	double fil_variance; // total variance
	double fil_std_dev;  // standard deviation derived from total variance
	double fil_total_mean;  // sum of channel means
	time_t utc_start;
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
// i think i can safely remove it  
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

	virtual double get_variance(void) { return fil_variance; }

	virtual void set_variance(double var) { this->fil_variance = var; }

	virtual double get_mean(void) { return fil_total_mean;}

	virtual void set_mean(double mean) { this->fil_total_mean = mean; }

	virtual double get_std_dev(void) { return fil_std_dev; }

	virtual void set_std_dev(double dev) { this->fil_std_dev = dev; }

	virtual time_t get_utc_start(void) { return utc_start; }
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


	const int seconds_in_day = 86400;
	double mjdstart = hdr.tstart;

	int days = (int)mjdstart;
	double fdays = mjdstart - (double)days;
	double seconds = fdays * (double)seconds_in_day;
	int secs = (int)seconds;
	double fracsec = seconds - (double)secs;
	if (fracsec - 1 < 0.0000001)
		secs++;

	int julian_day = days + 2400001;
	int n_four = 4  * (julian_day+((6*((4*julian_day-17918)/146097))/4+1)/2-37);
      	int n_dten = 10 * (((n_four-237)%1461)/4) + 5;

      	struct tm gregdate;
      	gregdate.tm_year = n_four/1461 - 4712 - 1900; // extra -1900 for C struct tm
      	gregdate.tm_mon  = (n_dten/306+2)%12;         // struct tm mon 0->11
      	gregdate.tm_mday = (n_dten%306)/10 + 1;

      	gregdate.tm_hour = secs / 3600;
      	secs -= 3600 * gregdate.tm_hour;

      	gregdate.tm_min = secs / 60;
      	secs -= 60 * (gregdate.tm_min);

      	gregdate.tm_sec = secs;

      	gregdate.tm_isdst = -1;
      	this->utc_start = mktime (&gregdate);


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


   this->nsamps = new_nsamples;
   this->tsamp  = new_tsamp;
   this->data   = data_new;

	// create vector fo keys which will be corresponding to channel numbers
	// keys_vector = 1, 2 ,3 , ... , 1024, 1, 2, 3, ... etc
	// then reduce by key, which will sum all the samples for a given channel

	// fiin lower nearest power of 2 to the new_nsamples

	size_t power_two_nsamples = 1 << (int)floor(log2((double)new_nsamples));


	// to avoid problems with memory process only a portion of timeseries at a time
	// divide into 512 chunks
	// no difference if 1/512th is processed or 1/16th
	// use 1/512th for efficiency purposes
	size_t to_process = power_two_nsamples / 512;

	unsigned char *timesamples_to_process = new unsigned char[to_process * nchans];
	std::copy(data_new, data_new + to_process * nchans, timesamples_to_process);
	int *keys_array = new int[to_process * nchans];

	std::generate_n(keys_array, to_process * nchans, channel_mod);

	double  *sum_array = new double[nchans];
	int *reduced_keys_array = new int[nchans];

	thrust::stable_sort_by_key(keys_array, keys_array + to_process * nchans,
					timesamples_to_process);

	// timesamples_to_process is now arranged by the channel number
	// starting with channel 1


	thrust::pair<int*,double*> keys_values;

	keys_values = thrust::reduce_by_key(keys_array, keys_array + to_process * nchans,
			timesamples_to_process, reduced_keys_array,
			sum_array);

	double *factor_array = new double[nchans];

	std::fill(factor_array, factor_array + nchans, (1.0/(double)to_process));

	double *mean_array = new double[nchans];

	thrust::transform(sum_array, sum_array + nchans, factor_array, mean_array,
				thrust::multiplies<double>());


	double *rms_array = new double[nchans];

	double *squared_timeseries = new double[to_process * nchans];

	thrust::transform(timesamples_to_process, timesamples_to_process + to_process * nchans,
				timesamples_to_process, squared_timeseries,
				thrust::multiplies<double>());



	double *squares_sum_array = new double[nchans];

	thrust::pair<int*,double*> rms_keys_values;
	rms_keys_values = thrust::reduce_by_key(keys_array, keys_array + to_process * nchans,
				squared_timeseries, reduced_keys_array,
				squares_sum_array);

	double *squares_mean = new double[nchans];

	thrust::transform(squares_sum_array, squares_sum_array + nchans, factor_array,
				squares_mean, thrust::multiplies<double>());


	std::cout << "Calculating variances..." << std::endl;


	// need to calculate variance
	// for each in timesamples_to_process, subtract mean for a given channel
	// then square and divide by the number of time samples processed

	double *channel_mean_timeseries = new double[to_process * nchans];
	double *sample_mean_diff = new double[to_process * nchans];
	double *sample_mean_diff_sqr = new double[to_process * nchans];
	double *diff_sqr_sum = new double[nchans];
	double *variance = new double[nchans];

	// expand the mean for the channel to all timesamples in a channel
	for(int m = 0; m < nchans; m++)
	{
		for(size_t n = 0; n < to_process; n++)
			channel_mean_timeseries[m*to_process + n] = mean_array[m];
	}

	// (x - <x>) step
	thrust::transform(timesamples_to_process, timesamples_to_process + nchans * to_process,
				channel_mean_timeseries, sample_mean_diff,
				thrust::minus<double>());
	// (x - <x>)^2 step
	thrust::transform(sample_mean_diff, sample_mean_diff + nchans * to_process,
				sample_mean_diff, sample_mean_diff_sqr,
				thrust::multiplies<double>());

	thrust::pair<int*,double*> difference_reduction;
	difference_reduction = thrust::reduce_by_key(keys_array, keys_array + to_process * nchans,
				sample_mean_diff_sqr, reduced_keys_array,
				diff_sqr_sum);

	thrust::transform(diff_sqr_sum, diff_sqr_sum + nchans,
				factor_array, variance, thrust::multiplies<double>());

	for(int i = 0; i < nchans; i++) rms_array[i] = sqrt(squares_mean[i]);

	std::ofstream means ("mean_values.dat", std::ofstream::out | std::ofstream::trunc);

	for(int j = 0; j < nchans; j++) means <<  j+1 << " " << (int)mean_array[j] 
						<< " " << (int)rms_array[j]
						<< " " << (int)variance[j] << endl;

	delete[] rms_array;

	double channels_sum;
	double channels_mean;
	double channels_variance;

	channels_sum = thrust::reduce(mean_array, mean_array + nchans, 0);

	channels_mean = channels_sum / (double)1024;

	double *mean_expand = new double[nchans];
	double *channels_mean_diff = new double[nchans];
	double *channels_mean_diff_sqr = new double[nchans];

	std::fill(mean_expand, mean_expand + nchans, channels_mean);

	thrust::transform(mean_array, mean_array + nchans, mean_expand,
				channels_mean_diff, thrust::minus<double>());

	thrust::transform(channels_mean_diff, channels_mean_diff + nchans,
				channels_mean_diff, channels_mean_diff_sqr,
				thrust::multiplies<double>());

	channels_variance = thrust::reduce(channels_mean_diff_sqr,
		channels_mean_diff_sqr + nchans, 0);

	channels_variance /= (double)1024;

	means << endl << endl << "Channels mean: " << channels_mean
		<< endl << "Channels variance: " << channels_variance;

	thrust::plus<double> binary_op;

	double total_variance = thrust::reduce(variance, variance + nchans, 0.0, binary_op);
	double total_mean = thrust::reduce(mean_array, mean_array + nchans, 0.0, binary_op);
	double standard_dev = sqrt(total_variance);


	means << endl << endl << "Total variance: " << total_variance
		<< endl << "Mean sum: " << total_mean
		<< endl << "Standard deviation: " << standard_dev;

	means.close();

	fil_total_mean = total_mean;
	fil_variance = total_variance;
	fil_std_dev = standard_dev;

	// NO NEED FOR COVARIANCE MATRIX NOW - USE WHEN STUFF GOES WRONG
/*
	double *covariance_matrix = new double[nchans * nchans];
	double *sample_mean_diff_trans = new double[nchans * to_process];
	bool *mask_array = new bool[nchans * nchans];

	// obtain covariance matrix
	// the result will be a 1024 x 1024 matrix
	std::cout << "Calculating covariance matrix..." << std::endl;

	// need to sum up (x - <x>) for each channel

	// have a matrix of x - <x>
	// multiply by its transpose

	// create transpose
	for (int i = 0; i < to_process; i++)
	{
		for (int j = 0; j < nchans; j++)
		{
			sample_mean_diff_trans[j + i * nchans] =
				sample_mean_diff[i + j * to_process];
		}
	}

	//multiplication
	double sum = 0.0;
	int progress = 0;


	for (int i = 0; i < nchans; i++)
	{
		for (int j = 0; j < nchans ; j++)
		{
			for (int k = 0; k < to_process ; k++)
			{
				sum += sample_mean_diff[i * to_process + k] * 
					sample_mean_diff_trans[j + k * nchans];
			}
			covariance_matrix[i * nchans + j] = sum / (double)to_process;
			if (covariance_matrix[i * nchans + j] >= 0.5)
				mask_array[i * nchans + j] = 1;
			else mask_array[i * nchans + j] = 0;
			sum = 0.0;
		}
		progress = (int)((double)i/(double)nchans * (double)100);
		// \r returns to the start of the line
		std::cout << "Completed " << progress << "% of covariance matrix\r";
		std::cout.flush(); // print of immediately without buffering
	}

	std::cout << std::endl;
	std::ofstream cov_file("covariance_matrix.dat", std::ofstream::out | std::ofstream::trunc);

	for (int i = 0; i < nchans; i++)
	{
		for (int j = 0; j < nchans; j++)
		{
			cov_file << (int)covariance_matrix[i * nchans + j] << " "; 
		}
		cov_file << std::endl;
	}

	cov_file.close();

	std::ofstream mask_file("mask_covariance.dat", std::ofstream::out | std::ofstream::trunc);

	for (int i = 0; i < nchans; i++)
	{
		for (int j = 0; j < nchans; j++)
		{
			mask_file << mask_array[i * nchans + j] << " ";
		}
		mask_file << std::endl;
	}

*/

	delete[] squares_mean;
	delete[] squares_sum_array;
	delete[] squared_timeseries;
	delete[] mean_array;
	delete[] factor_array;
	delete[] reduced_keys_array;
	delete[] sum_array;
	delete[] timesamples_to_process;
	delete[] sample_mean_diff;
	delete[] sample_mean_diff_sqr;
	delete[] diff_sqr_sum;
	delete[] variance;
	delete[] mean_expand;
	delete[] channels_mean_diff;
	delete[] channels_mean_diff_sqr;
//	delete[] covariance_matrix;
//	delete[] sample_mean_diff_trans;
//	delete[] mask_array;

//	std::vector<unsigned char> timesamples_to_process(data_new,
//					data_new + to_process * nchans);

//	std::vector<unsigned char> sorted_timesamples(to_process * nchans);

//	std::vector<int> keys_vector(to_process*nchans);
//	std::vector<size_t> sum_vector(nchans);
//	std::vector<int> reduced_keys_vector(nchans);
//	std::generate_n(keys_vector.begin(), to_process*nchans, channel_mod);

//	thrust::stable_sort_by_key(keys_vector.begin(), keys_vector.end(),
//					sorted_timesamples.begin());
//	thrust::pair<std::vector<int>::iterator,std::vector<size_t>::iterator> keys_values;
//	keys_values = thrust::reduce_by_key(keys_vector.begin(), keys_vector.end(),
//				sorted_timesamples.begin(), reduced_keys_vector.begin(),
//				sum_vector.begin());


//	cout << "Testing the keys_vector: ";
//	for (int i = 0; i < 2048; i++) cout << keys_vector[i] << " ";

//	std::cin.get();

//	thrust::reduce_by_key(

//	std::cout << "Printing some results\n";
// 	std::cout << (int)data_new[0] << " " << (int)data_temp[0] <<  " " << (int)data_temp[hdr.nchans] << std::endl;

//   	std::cout << (int)this->data[0] << std::endl;
//  	std::cout << this->tsamp << std::endl;

	delete [] data_temp;		// cleaning


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
