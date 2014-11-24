#include <cmath>
#include <data_types/filterbank.hpp>

int current = 0;

int channel_mod () { current++; int modulus = current % 1024; if (modulus == 0) return 1024;
                                else return modulus;} 

