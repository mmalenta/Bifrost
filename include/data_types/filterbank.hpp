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

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
using std::cin;
using std::cout;
using std::endl;

extern int current;
int channel_mod (); // { current++; int modulus = current % 1024; if (modulus == 0) return 1024;
		//		else return modulus;}

// following functors will work correctly on device - no data races
class Channel_mod_dev
{
        public:
               	__host__ __device__ Channel_mod_dev(unsigned int channels) :
                                        n_channels(channels) {};
                __host__ __device__ double operator() (double el_index)
                        {
                                double modulus = (size_t)el_index % (size_t)n_channels;
                                if (modulus == 0) return n_channels;
                                        else return modulus;
                        }
        private:
                unsigned int n_channels;
};

class Expand_mean_dev
{
        public:
               	__host__ __device__ Expand_mean_dev(double *means, size_t chunk) :
                                        my_mean(means), my_chunk(chunk) {};
                __host__ __device__ double operator() (double el_index)
                        {
                                int mean_idx = (size_t)el_index / my_chunk;
                                return my_mean[mean_idx];
                        }
        private:
                double* my_mean;
                size_t my_chunk;
};

class Timesample_index
{
       	public:
               	__host__ __device__ Timesample_index(int chans) : chans_no(chans) {};

                __host__ __device__ int operator() (int el_index)
                       	{
                               	int samp_idx = (el_index / chans_no) + 1;
                               	return samp_idx;
                        }
        private:
                size_t chans_no;
};

class Timesample_index_final
{
        public:
               	__host__ __device__ Timesample_index_final(void) {};

                __host__ __device__ size_t operator() (size_t index, size_t seq)
                        {
                                double final_idx = (double)index - (double)seq;

                                final_idx = final_idx + abs(final_idx);

                                final_idx /= 2;

                                return (size_t)final_idx;
                        }
};

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
	double *mean_array;	// means for different bins
	double *stdev_array;	// standard deviations for different beans
	unsigned int bins;	// the number of bins used to split the time series (might not use)
  	unsigned int nsamps; /*!< Number of time samples. */
	unsigned int chunk_nsamps; // number of time samples in the chunk
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

	virtual unsigned int get_chunk_nsamps(void){return chunk_nsamps;}
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

	virtual double* get_mean_array(void){return this->mean_array;}

	virtual double* get_stdev_array(void){return this->stdev_array;}

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
  SigprocFilterbank(std::string filename, unsigned int disp_diff=0)
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

/*	cout << *data_temp << " " << *(data_temp + 1) << " " << *(data_temp + 2)
                << " " << *(data_temp + hdr.nchans) << endl;

	cout << (unsigned char)*data_temp << " " << (unsigned char)*(data_temp + 1) << " " << (unsigned int)*(data_temp + 2)
                << " " << (unsigned int)*(data_temp + hdr.nchans) << endl;

	unsigned int sum = (unsigned int)(*data_temp + *(data_temp + hdr.nchans));

	unsigned char sum_char = *data_temp + *(data_temp+ hdr.nchans);

	cout << (unsigned char)*data_temp << " "
		<< (unsigned int)(*data_temp + *(data_temp + hdr.nchans)) << " "
		<< sum << " " << (unsigned char)sum << endl;

	cout << sum_char << " " << (unsigned int) sum_char;

	cin.get();
*/
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
    	std::cout << "The channel bandwidth [MHz]: " << hdr.foff << std::endl;
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

	// move into array
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

	std::cout << "Finished averaging time samples!\n";

	// create vector fo keys which will be corresponding to channel numbers
        // keys_vector = 1, 2 ,3 , ... , 1024, 1, 2, 3, ... etc
        // then reduce by key, which will sum all the samples for a given channel

        // fiin lower nearest power of 2 to the new_nsamples

        size_t power_two_nsamples = 1 << (int)floor(log2((double)new_nsamples));

	unsigned int chunks_no = 512;

        size_t to_process = power_two_nsamples;
        size_t data_chunk = power_two_nsamples/chunks_no; // used to process different means

        cout << "Will process " << to_process << " time samples" << endl;
        cout << "in chunks of " << data_chunk << " samples" << endl;
        cout << "spanning " << (double)data_chunk * new_tsamp << " seconds" << endl;

	// not sure what to do if it doesn't - will worry about it later
	if((new_nsamples - power_two_nsamples) > 4 * (nchans - 1))
		std::cout << "Will use crude dedispersion for statistics" << endl;

	this->chunk_nsamps = data_chunk;

	unsigned int nchans_less = nchans - 1;
	unsigned int diff_per_chan = disp_diff;	// "disperse" time series in steps of diff_per_chan per channel
	size_t total_more = diff_per_chan * nchans_less;	// total number of extra time samples
	size_t total_samps = data_chunk + total_more;

	cout << "Dispersion per channel difference " << diff_per_chan << " time samples\n";

	// Thrust vectors automatically deleted when the function returns
        thrust::equal_to<int> binary_pred;
        thrust::plus<double> binary_op;

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start,0);

        // "dedispersing" might slow the things down as we are processing much larger chunks
        thrust::device_vector<double> d_chunk_to_process(data_chunk * nchans + total_more * nchans);
        thrust::device_vector<int> d_reduced_chunk_keys(data_chunk + total_more);
        thrust::device_vector<double> d_chunk_timesamples_double(data_chunk);
        thrust::device_vector<double> d_single_timesample(nchans);
        thrust::device_vector<double> d_chunk_timesamples_full(data_chunk + total_more);                // timesamples summed over all channels
        thrust::device_vector<double> d_chunk_timesamples(data_chunk);
        thrust::device_vector<double> d_chunk_timesamples_diff(data_chunk);
        thrust::device_vector<double> d_chunk_timesamples_diff_sqr(data_chunk);
        thrust::device_vector<size_t> d_timesample_index(data_chunk * nchans + total_more * nchans);
        thrust::device_vector<double> d_chunk_mean(data_chunk);


/*        thrust::device_vector<double> d_chunk_to_process(data_chunk * nchans);
        thrust::device_vector<double> d_chunk_keys_array(data_chunk * nchans);
        thrust::device_vector<double> d_reduced_chunk_keys_array(nchans);
        thrust::device_vector<double> d_chunk_sum_array(nchans);
        thrust::device_vector<double> d_chunk_mean_array(nchans);
        thrust::device_vector<double> d_chunk_mean_expand_array(data_chunk * nchans);
        thrust::device_vector<double> d_chunk_diff_array(data_chunk * nchans);
        thrust::device_vector<double> d_chunk_diff_sqr_array(data_chunk * nchans);
        thrust::device_vector<double> d_chunk_var_array(nchans);

        thrust::device_vector<double> d_full_chunk_mean(512);
        thrust::device_vector<double> d_full_chunk_var(512);
*/
/*      int *reduced_chunk_keys_array = new int[nchans];
        double *chunk_sum_array = new double[nchans];
        double *chunk_mean_array = new double[nchans];
        double *chunk_mean_expand_array = new double[data_chunk * nchans];
        double *chunk_diff_array = new double[data_chunk * nchans];
        double *chunk_diff_sqr_array = new double[data_chunk * nchans];
        double *chunk_var_array = new double[nchans];
*/
        thrust::constant_iterator<double> rec_chunk_start(1.0/(double)data_chunk);
        thrust::constant_iterator<double> rec_chunk_end = rec_chunk_start + nchans;

        double *full_chunk_mean = new double[chunks_no];
        double *full_chunk_var = new double[chunks_no];
        double *full_chunk_std = new double[chunks_no];

	// device iterators for thrust::pair
        typedef thrust::device_vector<int>::iterator intIter;
        typedef thrust::device_vector<double>::iterator doubIter;

        // generate device_vector used for index transformations

        thrust::device_vector<size_t> d_sequence_chunk(nchans);
        thrust::device_vector<size_t> d_total_seq(data_chunk * nchans + total_more * nchans);

	if (diff_per_chan !=0)
	{
		thrust::sequence(d_sequence_chunk.begin(), d_sequence_chunk.end(), (size_t)0, (size_t)diff_per_chan);

        	for (int time_samp = 0; time_samp < total_samps; time_samp++)
               		thrust::copy_n(d_sequence_chunk.begin(), nchans, d_total_seq.begin() + time_samp * nchans);
	} else
	{
		thrust::fill(d_total_seq.begin(), d_total_seq.end(), (size_t)0);
	}
        // will need to copy diff_per_chan * (nchans - 1) time samples extra
        // that is diff_per_chan * (nchans - 1) * nchans more in total


        for (int chunk_no = 0; chunk_no < chunks_no; chunk_no++)
        {

                int chunk_no_extra = chunk_no + 1;

                cout << "Processing chunk number " << chunk_no << "\r";
		cout.flush();
                // if (chunk_no + 1) used - treated as type casting

		// copy extra 4 * (nchans - 1) time samples to accommodate "dedispersion"
		// of the last time sample in the original data chunk

//              	thrust::copy(data_new + chunk_no * data_chunk * nchans,
//                                data_new + chunk_no_extra * data_chunk * nchans + 4 * nchans_less,
//                                d_chunk_to_process.begin());

		thrust::copy_n(data_new + chunk_no * data_chunk * nchans,
				total_samps * nchans,
				d_chunk_to_process.begin());

                thrust::device_ptr<double> sample_ptr = d_chunk_to_process.data();

		// "DEDISPERSION ALGORITHM"

                // INDEX TRANSFORMATIONS
                // need to do this sequence for every chunk as it gets sorted by key
                thrust::sequence(d_timesample_index.begin(), d_timesample_index.end(), (size_t)0, (size_t)1);

                thrust::transform(d_timesample_index.begin(),
                                        d_timesample_index.end(),
                                        d_timesample_index.begin(),
                                        Timesample_index(nchans));

                thrust::transform(d_timesample_index.begin(),
                                        d_timesample_index.end(),
                                        d_total_seq.begin(),
                                        d_timesample_index.begin(),
                                        Timesample_index_final());

                thrust::pair<intIter,doubIter> chunk_keys_values;

                thrust::sort_by_key(d_timesample_index.begin(),
                                        d_timesample_index.end(),
                                        d_chunk_to_process.begin());

                chunk_keys_values = thrust::reduce_by_key(d_timesample_index.begin(),
                                                                d_timesample_index.end(),
                                                                d_chunk_to_process.begin(),
                                                                d_reduced_chunk_keys.begin(),
                                                                d_chunk_timesamples_full.begin(),
                                                                binary_pred,
                                                                binary_op);

		if (diff_per_chan != 0)
		{
                	thrust::copy_n(d_chunk_timesamples_full.begin() + 1,
					data_chunk,
					d_chunk_timesamples.begin());
		} else
		{
			thrust::copy_n(d_chunk_timesamples_full.begin(),
					data_chunk,
					d_chunk_timesamples.begin());
		}

                full_chunk_mean[chunk_no] = thrust::reduce(d_chunk_timesamples.begin(),
                                                                d_chunk_timesamples.end(),
                                                                (double)0.0,
                                                                binary_op) / data_chunk;

                thrust::constant_iterator<double> channels_mean(full_chunk_mean[chunk_no]);

                thrust::fill(d_chunk_mean.begin(), d_chunk_mean.end(), full_chunk_mean[chunk_no]);

		                thrust::transform(d_chunk_timesamples.begin(),
                                        d_chunk_timesamples.end(),
                                        d_chunk_mean.begin(),
                                        d_chunk_timesamples_diff.begin(),
                                        thrust::minus<double>());

                thrust::transform(d_chunk_timesamples_diff.begin(),
                                        d_chunk_timesamples_diff.end(),
                                        d_chunk_timesamples_diff.begin(),
                                        d_chunk_timesamples_diff_sqr.begin(),
                                        thrust::multiplies<double>());

                full_chunk_var[chunk_no] = thrust::reduce(d_chunk_timesamples_diff_sqr.begin(),
                                                                d_chunk_timesamples_diff_sqr.end(),
                                                                0.0,
                                                                binary_op) / (double)data_chunk;

                full_chunk_std[chunk_no] = sqrt(full_chunk_var[chunk_no]);

	}
	//const double first_mean_diff = front_end_mean_diff[0];

	//for (int chunk_no = 0; chunk_no < 512; chunk_no++)
        //        front_end_mean_diff[chunk_no] = first_mean_diff - front_end_mean_diff[chunk_no];


	//fil_total_mean = full_chunk_mean[0] * nchans;

	cout <<  "Finished processing all chunks...\n";
	cout << "Will now save\n";

        std::ofstream means_out("means_values_dev.dat", std::ofstream::out | std::ofstream::trunc);

        for (int chunk_no = 0; chunk_no < chunks_no; chunk_no++)
                means_out << chunk_no << " " << full_chunk_mean[chunk_no] << " "
                                << full_chunk_var[chunk_no] << " "
                                << full_chunk_std[chunk_no] <<  endl;

        means_out.close();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed;

        cudaEventElapsedTime(&elapsed, start, stop);

        cout << "Time taken to go through all chunks: " << elapsed / 1000.0f << "s\n";

        cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// FINISHED MEAN AND STANDARD DEVIATION PART
	// SMOOTHING PART START HERE

        cout << "Smoothing the data\n";

        // REMEMBER -- WEIGHTS MUST ADD UP TO 1!!!
        double *smoothing_function = new double[32];

        thrust::device_vector<double> d_smoothing_function(32);
        thrust::device_vector<double> d_mean(chunks_no);
        thrust::device_vector<double> d_variance(chunks_no);
        thrust::device_vector<double> d_stdev(chunks_no);
        thrust::device_vector<double> d_mean_smoothed_part(32);
        thrust::device_vector<double> d_stdev_smoothed_part(32);

        std::ifstream func_in("smoothing_function.dat");

        if (!func_in.is_open())
        {
                cout << "Could not open the file, will use simple weighted mean as smoothing\n";
                for (int i = 0; i < 32; i++)
			smoothing_function[i] = (double)1.0/(double)32.0;
        } else
	{
                for(int i = 0; i < 32; i++)
                        func_in >> smoothing_function[i];
        }

	func_in.close();

	thrust::copy(smoothing_function, smoothing_function + 32, d_smoothing_function.begin());

        //this is a very simple smoothing function -- box with weight of 1/16
        //thrust::fill(d_smoothing_function.begin(),
        //              d_smoothing_function.end(),
        //              (double)1.0/(double)32.0);

        thrust::copy(full_chunk_mean, full_chunk_mean + chunks_no, d_mean.begin());
        thrust::copy(full_chunk_var, full_chunk_var + chunks_no, d_variance.begin());
        thrust::copy(full_chunk_std, full_chunk_std + chunks_no, d_stdev.begin());

        double *full_chunk_mean_smooth = new double[chunks_no];
        double *full_chunk_var_smooth = new double[chunks_no];
        double *full_chunk_std_smooth = new double[chunks_no];

        thrust::device_ptr<double> mean_ptr = d_mean.data();
        thrust::device_ptr<double> stdev_ptr = d_stdev.data();

        thrust::plus<double> sum_op;

        for (int smooth_start = 0; smooth_start < 16; smooth_start++)
        {
                full_chunk_mean_smooth[smooth_start] = thrust::reduce(mean_ptr,
                                                        mean_ptr + smooth_start +16,
                                                        0.0, sum_op) / (double)((double)smooth_start + 16.0);
                full_chunk_std_smooth[smooth_start] = thrust::reduce(stdev_ptr,
                                                        stdev_ptr + smooth_start + 16,
                                                        0.0, sum_op) / (double)((double)smooth_start + 16.0);
        }

        for (int smooth_start = chunks_no - 16; smooth_start < chunks_no; smooth_start++)
        {
                full_chunk_mean_smooth[smooth_start] = thrust::reduce(mean_ptr + smooth_start - 16,
                                                        mean_ptr + chunks_no,
                                                        0.0, sum_op) / (double)(chunks_no - (double)smooth_start + 16.0);
                full_chunk_std_smooth[smooth_start] = thrust::reduce(stdev_ptr + smooth_start - 16,
                                                        stdev_ptr + chunks_no,
                                                        0.0, sum_op) / (double)(chunks_no - (double)smooth_start + 16.0);
        }

	for (int smooth_start = 16; smooth_start < chunks_no - 16; smooth_start++)
        {
                thrust::transform(mean_ptr + smooth_start - 16,
                                        mean_ptr + smooth_start + 16,
                                        d_smoothing_function.begin(),
                                        d_mean_smoothed_part.begin(), thrust::multiplies<double>());

                thrust::transform(stdev_ptr + smooth_start - 16,
                                        stdev_ptr + smooth_start + 16,
                                        d_smoothing_function.begin(),
                                        d_stdev_smoothed_part.begin(), thrust::multiplies<double>());
                full_chunk_mean_smooth[smooth_start] = thrust::reduce(d_mean_smoothed_part.begin(),
                                                                d_mean_smoothed_part.end(),
                                                                0.0, sum_op);

                full_chunk_std_smooth[smooth_start] = thrust::reduce(d_stdev_smoothed_part.begin(),
                                                                d_stdev_smoothed_part.end(),
                                                                0.0, sum_op);
	}

	        std::ofstream smooth_out("means_values_smooth_full.dat",
                                        std::ofstream::out | std::ofstream::trunc);

        for (int chunk_idx = 0; chunk_idx < chunks_no; chunk_idx++)
                smooth_out << chunk_idx << " " << full_chunk_mean_smooth[chunk_idx]
                        << " " << full_chunk_std_smooth[chunk_idx] << endl;
/*
        delete [] chunk_to_process;
        delete [] chunk_keys_array;
        delete [] reduced_chunk_keys_array;
        delete [] chunk_sum_array;
        delete [] chunk_mean_array;
        delete [] full_chunk_mean;
        delete [] chunk_mean_expand_array;
        delete [] chunk_diff_array;
        delete [] chunk_diff_sqr_array;
        delete [] chunk_var_array;
*/

//	for (int chunk_no = 0; chunk_no < 512; chunk_no++)
//		full_chunk_mean_smooth[chunk_no] = full_chunk_mean_smooth[chunk_no] * (double)1024.0 +
//							front_end_mean_diff[chunk_no] * (double)1024.0;

//	cout << fil_total_mean << " " << full_chunk_mean_smooth[0];

//	cin.get();

	this->mean_array = full_chunk_mean_smooth;
	this->stdev_array = full_chunk_std_smooth;


	// turn smoothing off
//	this->mean_array = full_chunk_mean;
//	this->stdev_array = full_chunk_std;

	delete [] full_chunk_mean;
	delete [] full_chunk_var;
	delete [] full_chunk_std;

//	delete [] full_chunk_mean_smooth;
//	delete [] full_chunk_var_smooth;
//	delete [] full_chunk_std_smooth;

	// OLD CODE STARTS BELOW


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

	delete [] data_temp;		// cleaning



   }


  /*!
    \brief Deconstruct a SigprocFilterbank object.
    The deconstructor cleans up memory allocated when
    reading data from file.
  */
~SigprocFilterbank()
  {
    	delete [] this->data;
//	delete [] this->mean_array;
//	delete [] this->stdev_array;
}
};
