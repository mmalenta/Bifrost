#!/usr/bin/env python 

import sys, os, tempfile, time
import numpy as np
import trans_paths

DM_MAX = 2000
# TODO check what these mean
TM = 0.06
BM = 0.14
LM = 0.06
RM = 0.06
#DX = 0.06
#DY = 0.14
NX = 2
NY = 2
SX = 1./NX - (LM + RM)
SY = 1./NY - (TM + BM)

class SNRDMPlot(object):
    def __init__(self, g):
        self.g = g
        self.dm_max = DM_MAX
        
    def plot(self, data):
        self.g.reset()
        self.g('set autoscale x')
        self.g('set xlabel "DM [pc cm^{-3}]"')
        self.g('set ylabel "SNR (approx)"')
        self.g('set ytics auto')
        self.g('set grid noxtics nomxtics ytics mytics lt 9 lw 0.2')
        self.g('PX = 1')
        self.g('PY = 1')
        self.g('set size SX, SY')
        self.g('set origin PX*(SX+LM+RM) + LM, PY*(SY+TM+BM) + BM')
        self.g('plot "' + data + '" u 2:3 w l notitle')

class SNRTimePlot(object):
    def __init__(self, g):
        self.g = g

    def plot(self, data):
        self.g.reset()
        self.g('PX = 0')
        self.g('PY = 0')
        self.g('dx = 0.0')
        self.g('dy = 0.0')
        self.g('sx = SX')
        self.g('sy = SY/4')
        self.g('set bmargin dy; set tmargin dy; set lmargin dx; set rmargin dx')
        self.g('set size sx, sy')

        self.g('set autoscale x')
        self.g('set xlabel "Time [s]"')
        self.g('set ylabel "SNR"')
        self.g('unset key')
        self.g('set ytics 2 format ""')
        self.g('set grid noxtics nomxtics ytics mytics lt 9 lw 0.2')

        self.g('set origin PX*(SX+LM+RM) + LM, PY*(SY+TM+BM) + BM + 0*sy')
        self.g('set ylabel "DM 0"')
        self.g('plot 0 w l lt 9, "' + data +'" u 2:6 w l lt 1')

        self.g('set xlabel ""')
        self.g('set xtics format ""')
        self.g('set origin PX*(SX+LM+RM) + LM, PY*(SY+TM+BM) + BM + 1*sy')
        self.g('set ylabel "DM-1"')
        self.g('plot 0 w l lt 9, "' + data +'" u 2:4 w l lt 1')

        self.g('set origin PX*(SX+LM+RM) + LM, PY*(SY+TM+BM) + BM + 2*sy')
        self.g('set ylabel "SNR"')
        self.g('plot 0 w l lt 9, "' + data +'" u 2:3 w l lt 1')

        self.g('set origin PX*(SX+LM+RM) + LM, PY*(SY+TM+BM) + BM + 3*sy')
        self.g('set ylabel "DM+1"')
        self.g('plot 0 w l lt 9, "' + data +'" u 2:5 w l lt 1')


class FreqTimePlot(object):
    def __init__(self, g, dm, in_nsamps, tsamp, f0, f1):
        self.g = g
        self.dm = dm
        self.in_nsamps = in_nsamps
        self.tsamp = tsamp
        self.f0 = f0
        self.f1 = f1
    def plot(self, data):
        self.g.reset()
        self.g('PX = 1')
        self.g('PY = 0')
        self.g('set size SX, SY')
        self.g('set origin PX*(SX+LM+RM) + LM, PY*(SY+TM+BM) + BM')

        self.g('set autoscale x')
        self.g('set autoscale y')
        self.g('unset colorbox')
        self.g('unset xtics')
        self.g('unset ytics')
        self.g('unset x2tics')
        self.g('unset y2tics')
        self.g('set xlabel "Time"')
        self.g('set ylabel "Frequency"')

        if self.in_nsamps > 0:
          in_nsamps = self.in_nsamps
        else:
          in_nsamps = 4096

        in_ntime = tsamp * in_nsamps
        t0 = 0 - (in_ntime / 2.0)
        t1 = 0 + (in_ntime / 2.0)
        self.g('t0 = ' + str(t0))
        self.g('t1 = ' + str(t1))
        self.g('f0 = ' + str(f0))
        self.g('f1 = ' + str(f1))
        self.g('dm = ' + str(dm))
        self.g('k = 4.148808e3')
        self.g('g(x) = sqrt(1. / (x / (dm*k) + 1./f0**2))')

        self.g('set x2range [t0:t1]')
        self.g('set y2range [f0:f1]')

        # TODO: Work out equation for DM curve in pixel coords
        self.g('plot "' + data + '" matrix with image notitle, g(x) w l notitle lw 2 lc 2 axes x2y2')


if __name__ == "__main__":
    import argparse
    import Gnuplot
    
    parser = argparse.ArgumentParser(description="Generates plots for Heimdall Candidates.")
    parser.add_argument('fil_file', help='2 bit SIGPROC filterbank file')
    parser.add_argument('sample', type=int, help='Sample number event was detected')
    parser.add_argument('filter', type=int, help='Filter number in which event was found')
    parser.add_argument('dm', type=float, help='DM of candidate event')
    parser.add_argument('-tsamp', type=float,  default=0.000064, help='Sampling time')
    parser.add_argument('-f0', type=float,  default=1581.0, help='Highest freq channel')
    parser.add_argument('-f1', type=float,  default=1181.0, help='Lowest freq channel')
    parser.add_argument('-resolution', default="1024x768")
    parser.add_argument('-std_out', help='Output image to stdout', action="store_true")
    parser.add_argument('-no_plot', help='Do not produce image, just process data', action="store_true")
    parser.add_argument('-interactive', action="store_true")
    parser.add_argument('-verbose', action="store_true")
    args = parser.parse_args()
   
    verbose = args.verbose
    fil_file = args.fil_file
    sample = args.sample
    filter = args.filter
    dm = args.dm
    tsamp = args.tsamp
    f0 = args.f0
    f1 = args.f1
    std_out = args.std_out
    no_plot = args.no_plot
    resolution = args.resolution
    interactive = args.interactive
    res_parts = resolution.split("x")
    if (len(res_parts) != 2):
      sys.stderr.write("ERROR: resolution must be of form 1024x768")
      sys.exit(1)

    res_x = res_parts[0]
    res_y = res_parts[1]

    # change to temp dir
    workdir = tempfile.mkdtemp()

    # check the DM2000 file exists
    dmlist = trans_paths.getConfigDir() + '/dm2000.list'

    try:
      fptr = open(dmlist, 'r')
    except IOError:
      print "ERROR: cannot open dmlist " + dmlist
      os.rmdir(workdir)
      sys.exit(1)
    else:
      # determine the dm index in the file that matches
      dm_idx = 0
      lines = fptr.readlines()
      for line in lines:
        line = line.strip()
        if float(line) > dm:
          break
        dm_idx += 1

      if verbose:
        sys.stderr.write("DM index: " + str(dm_idx) + " from DM=" + str(dm) + "\n")
    
    out_nsamp = 128
    out_dmcount = 32
    fscrunch = 16

    cmd = "cd "+workdir+"; " + trans_paths.getBinaryDir() + "/candidate_profiler " + \
          fil_file + " " + \
          dmlist + " " + \
          str(sample) + " " + \
          str(filter) + " " + \
          str(dm_idx) + " " + \
          str(out_nsamp) + " " + \
          str(out_dmcount) + " " + \
          str(fscrunch)
    if verbose:
      sys.stderr.write ( "Running candidate profiler: " + cmd + "\n")

    in_nsamps = 0
    sys.stderr.write ( cmd + "\n")
    p = os.popen(cmd)
    for line in p.readlines():
      line = line.strip()
      sys.stderr.write(line + "\n")
      parts = line.split("=")
      if (parts[0] == "in_nsamps"):
        in_nsamps = int(parts[1])
    p.close()

    # time.sleep(1)

    if not no_plot:    
      # Generate plots
      if verbose:
        sys.stderr.write ( "Generating plot...\n")
      g = Gnuplot.Gnuplot(debug=0)
      if not interactive:
        g('set terminal pngcairo enhanced font "arial,10" size ' + res_x + ', ' + res_y)
        if std_out:
          g('set output')
          if verbose:
            sys.stderr.write ( "Writing binary image data to STDOUT\n")
        else:
          g('set output "candidate_' + resolution + '.tmp.png"')
          if verbose:
            sys.stderr.write ( "Writing plots to candidate_" + resolution + ".tmp.png\n")
      else:
        g('set terminal x11')

      g('set multiplot')
      g('set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0')
      g('TM = 0.06')
      g('BM = 0.06')
      g('LM = 0.06')
      g('RM = 0.06')
      g('NX = 2')
      g('NY = 2')
      g('SX = 1./NX - (LM + RM)')
      g('SY = 1./NY - (TM + BM)')

      snrdm_plot = SNRDMPlot(g)
      snrtime_plot = SNRTimePlot(g)
      freqtime_plot = FreqTimePlot(g, dm, in_nsamps, tsamp, f0, f1)

      snrdm_plot.plot(workdir + "/snr_dm.dat")
      snrtime_plot.plot(workdir + "/snr_time.dat")
      freqtime_plot.plot(workdir + "/freq_time.dat")
      
      g('unset multiplot')

    if interactive:
      raw_input('Please press return to close...\n')

    g.close()
    time.sleep(0.1)

    if verbose:
      sys.stderr.write ( "Files written to " + workdir + "\n")
    os.remove(workdir + "/snr_dm.dat")
    os.remove(workdir + "/snr_time.dat")
    os.remove(workdir + "/freq_time.dat")
    os.remove(workdir + "/dm_time.pgm")
    os.rmdir(workdir)

    if verbose:
      sys.stderr.write ( "Done\n")
