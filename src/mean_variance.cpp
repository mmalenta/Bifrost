#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include <utils/mean_variance.hpp>

#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

using std::cout;
using std::endl;

void perform_tests(unsigned char data_new[], size_t new_samples, size_t dm_size)
{
	size_t power_two_nsamples = 1 << (int)floor(log2((double)new_samples));
        cout << "Nearest power of 2: " << power_two_nsamples;

	size_t to_process = power_two_nsamples;

	cout << "Will process a chunk of " << to_process << " time samples for "
		 << dm_size << " DM trials" << endl;

	unsigned char *timesamples_to_process = new unsigned char[to_process];
	double *mean_array = new double[to_process];
	double *sample_mean_diff = new double[to_process];
	double *sample_mean_diff_sqr = new double[to_process];
	double sum;
	double mean;
	double diff_sqr_sum;
	double variance;

	std::ofstream output_file("dedispersed_means.dat", std::ofstream::out | std::ofstream::trunc);


	for (size_t dm_index = 0; dm_index < dm_size; dm_index++)
	{
		size_t offset = dm_index * new_samples;

		// read part of the dedispersed timeseries
		std::copy(data_new + offset, data_new + offset + to_process,
				timesamples_to_process);

		// get the sum of the timesaples read
		sum = thrust::reduce(timesamples_to_process, timesamples_to_process + to_process,
			0);

		mean = sum/to_process;

		//expand the mean to the whole dm trial
		std::fill(mean_array, mean_array + to_process, mean);

		// x - <x> step for the variance calculation
		thrust::transform(timesamples_to_process, timesamples_to_process + to_process,
			mean_array, sample_mean_diff, thrust::minus<double>());

		// (x - <x>)^2 step for variance calculation
		thrust::transform(sample_mean_diff, sample_mean_diff + to_process,
			sample_mean_diff, sample_mean_diff_sqr, thrust::multiplies<double>());

		// sum( (x - <x>)^2 ) step for variance calculation
		diff_sqr_sum = thrust::reduce(sample_mean_diff_sqr, sample_mean_diff_sqr + to_process,
			0);

		// sample variance
		variance = diff_sqr_sum / to_process;

		output_file << dm_index << " " << mean << " " << variance << endl;

	}

	output_file.close();

	delete[] timesamples_to_process;
	delete[] mean_array;
	delete[] sample_mean_diff;
	delete[] sample_mean_diff_sqr;
}
