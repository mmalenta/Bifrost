#!/usr/bin/env python
#
# Filename: trans_cand_server.py
#
#  * listen on socket
#  * return plots as binary data over socket 

import Dada, Bpsr, threading, sys, time, socket, select, signal, traceback
import time, numpy, math, os, fnmatch, tempfile, Gnuplot, datetime
import trans_paths

DL = 0

###########################################################################

def plotCandDspsr(fil_file, sample, filter, dm):

  cand_time = (0.000064 * sample)
  cmd = "dmsmear -f 1382 -b 400 -n 1024 -d " + str(dm) + " -q 2>&1 "
  p = os.popen(cmd)
  cand_band_smear = p.readline().strip()
  p.close()
  Dada.logMsg(1, DL, "plotCandDspsr: cand_band_smear='" + cand_band_smear + "'")

  cand_filter_time = (2 ** filter) * 0.000064

  cand_smearing = float(cand_band_smear) + float(cand_filter_time)

  cand_start_time = cand_time - (1.0 * cand_smearing)

  cand_tot_time   = 2.0 * cand_smearing

  cmd = "dspsr " + fil_file + " -S " + str(cand_start_time) + \
        " -b 128" + \
        " -T " + str(cand_tot_time) + \
        " -c " + str(cand_tot_time) + \
        " -D " + str(dm) + \
        " -U 8" + \
        " 2>&1 | grep unloading | awk '{print $NF}'"

  # create a temporary working directory
  workdir = tempfile.mkdtemp()

  os.chdir(workdir)

  Dada.logMsg(1, DL, "plotCandDspsr: " + cmd)
  p = os.popen(cmd)
  response = p.readline().strip()
  p.close()

  archive = response + ".ar"
  count = 10 
  while ((not os.path.exists(archive)) and count > 0):
    Dada.logMsg(1, DL, "plotCandDspsr: archive file [" + archive + "] did not exist")
    time.sleep(1)
    count = count - 1

  binary_data = []

  title = "DM=" + str(dm) + " Length=" + str(cand_filter_time*1000) + "ms Epoch=" + str(cand_start_time)
  cmd = "psrplot -c above:l='" + title + "' -j 'zap chan 0-160' -j 'F 128' -c x:unit=ms -p freq+ ./" + archive + " -D -/PNG";
  Dada.logMsg(1, DL, "plotCandDspsr: " + cmd)
  p = os.popen(cmd)
  binary_data = p.read()
  p.close()
  
  if os.path.exists(archive):
    os.remove(archive)
  os.chdir ("/")
  os.rmdir(workdir)

  return binary_data

def plotCandidate(fil_file, sample, filter, dm):

  verbose = False
  res_x = '800'
  res_y = '600'
  resolution = res_x + 'x' + res_y

  # change to temp dir
  workdir = tempfile.mkdtemp()

  # check the DM file exists
  dmlist = DM_LIST

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

  out_nsamp = 64
  out_dmcount = 32
  fscrunch = 32

  os.chdir(workdir)

  cmd = "candidate_profiler " + \
        fil_file + " " + \
        dmlist + " " + \
        str(sample) + " " + \
        str(filter) + " " + \
        str(dm_idx) + " " + \
        str(out_nsamp) + " " + \
        str(out_dmcount) + " " + \
        str(fscrunch)

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


  # Generate plots
  if verbose:
    sys.stderr.write ( "Generating plot...\n")
  g = Gnuplot.Gnuplot(debug=0)

  g('set terminal pngcairo enhanced font "arial,10" size ' + res_x + ', ' + res_y)
  g('set output "candidate_' + resolution + '.tmp.png"')
  if verbose:
    sys.stderr.write ( "Writing plots to candidate_" + resolution + ".tmp.png\n")

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
  freqtime_plot = FreqTimePlot(g, dm, in_nsamps)

  snrdm_plot.plot(workdir + "/snr_dm.dat")
  snrtime_plot.plot(workdir + "/snr_time.dat")
  freqtime_plot.plot(workdir + "/freq_time.dat")

  g('unset multiplot')
  g.close()

  img_file = open(workdir + '/candidate_' + resolution + '.tmp.png')
  try:
    bytes_read = img_file.read()
    content = bytes_read
  finally:
    img_file.close()

  os.chdir('/')

  if verbose:
    sys.stderr.write ( "Files written to " + workdir + "\n")
  os.remove(workdir + "/snr_dm.dat")
  os.remove(workdir + "/snr_time.dat")
  os.remove(workdir + "/freq_time.dat")
  os.remove(workdir + "/dm_time.pgm")
  os.remove(workdir + '/candidate_' + resolution + '.tmp.png')
  os.rmdir(workdir)

  return content 

############################################################################### 
#
# main
#
if __name__ == "__main__":

  import argparse
  import Gnuplot


  parser = argparse.ArgumentParser(description="Plot a transient event to stdout")
  parser.add_argument("fil_file", help="filterbank file to process")
  parser.add_argument("sample", help="sample number of event start", type=int)
  parser.add_argument("dm", help="dm of event", type=float)
  parser.add_argument("filter", help="filter of event", type=int)

  parser.add_argument('-verbose', action="store_true")
  args = parser.parse_args()

  fil_file = args.fil_file
  sample = args.sample
  dm = args.dm
  filter = args.filter
  verbose = args.verbose

  if not os.path.exists(fil_file):
    sys.exit('ERROR: Filterbank file %s was not found!' % fil_file)

  proc_type = "dspsr"

  try:
    fptr = open(fil_file, 'r')
  except IOError:
    Dada.logMsg(1, DL, "main: could not open filterbank file for reading")
    fil_file = ""
  else:
    fptr.close()

  if len(fil_file) == 0:
    Dada.logMsg(1, DL, "main: could not find filterbank file")
  else:
    binary_data = []

    if (proc_type == "dspsr"):
      Dada.logMsg(2, DL, "main: plotCandDspsr()")
      binary_data = plotCandDspsr(fil_file, sample, filter, dm)
      binary_len = len(binary_data)
      Dada.logMsg(3, DL, "main: sending binary data len="+str(binary_len))

    else:
      Dada.logMsg(2, DL, "main: plotCandidate()")
      binary_data = plotCandidate(fil_file, sample, filter, dm)
      binary_len = len(binary_data)
      Dada.logMsg(3, DL, "main: sending binary data len="+str(binary_len))

  sys.stdout.write(binary_data)

  # exit
  sys.exit(0)
