/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __DataSource_h
#define __DataSource_h

#define HD_TIMESTR "%Y-%m-%d-%H:%M:%S"

class DataSource {

  public:

    DataSource ()
    {
      nchan = 0;
      nbit  = 0;
      npol  = 0;
      stride = 0;
      beam = 0;
      utc_start = 0;
      tsamp = 0;
      spectra_rate= 0;
    }
    virtual ~DataSource() {}

    size_t get_nchan() const { return nchan; }
    size_t get_nbit()  const { return nbit; }
    size_t get_stride() const { return stride; }
    size_t get_beam()   const { return beam; }
    time_t get_utc_start()  const { return utc_start; }
    double get_spectra_rate() const  { return spectra_rate; }
    float  get_tsamp()  const { return tsamp; }
    float  get_f0()  const { return f0; }
    float  get_df()  const { return df; }

    // must be implemented in sub class
    virtual bool   get_error() const = 0;
    virtual size_t get_data(size_t nsamps, char* data) = 0;

    static time_t mjd2utctm (double mjd)
    {
      const int seconds_in_day = 86400;
      int days = (int) mjd;
      double fdays = mjd - (double) days;
      double seconds = fdays * (double) seconds_in_day;
      int secs = (int) seconds;
      double fracsec = seconds - (double) secs;
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
      time_t date = mktime (&gregdate);

      return date;
    }

  protected:
    size_t nchan;
    size_t nbit;
    size_t npol;
    size_t stride;
    size_t beam;
    time_t utc_start;
    float  tsamp;
    double f0;
    double df;
    double spectra_rate;

};

#endif
