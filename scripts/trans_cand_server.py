#!/usr/bin/env python
#
# Filename: trans_cand_server.py
#
#  * listen on socket
#  * return plots as binary data over socket 

LISTEN_HOST = "hipsr7.hipsr.local"
LISTEN_PORT = 55555
FILTERBANK_DIR = "/nfs/raid0/bpsr/perm"

import Dada, Bpsr, threading, sys, time, socket, select, signal, traceback
import time, numpy, math, os, fnmatch, tempfile, Gnuplot, datetime
import trans_paths

DM_LIST  = trans_paths.getConfigDir() + '/dm2000.list'
PIDFILE  = "trans_cand_server.pid"
LOGFILE  = "trans_cand_server.log"
QUITFILE = "trans_cand_server.quit"
DL = 2


###########################################################################

def signal_handler(signal, frame):
  print 'You pressed Ctrl+C!'
  global quit_event
  quit_event.set()


class SNRDMPlot(object):
    def __init__(self, g):
        self.g = g

    def plot(self, data):
        #self.g.reset()
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

        # determine min / max values for each column in file data
        snr_data = numpy.loadtxt(data,
                   dtype={'names': ('samp_idx','time','snr',
                                    'snr_dm-1','snr_dm+1','snr_dm0'),
                          'formats': ('i4', 'f4', 'f4', 'f4', 'f4', 'f4')})
        ymax = 0
        ymin = 0
        for row in snr_data:
          if (row[2] > ymax):
            ymax = row[2]
          if (row[2] < ymin):
            ymin = row[2]
          if (row[3] > ymax):
            ymax = row[3]
          if (row[3] < ymin):
            ymin = row[3]
          if (row[4] > ymax):
            ymax = row[4]
          if (row[4] < ymin):
            ymin = row[4]
          if (row[5] > ymax):
            ymax = row[5]
          if (row[5] < ymin):
            ymin = row[5]

        print "ymin = " + str(ymin) + " ymax=" + str(ymax)
        if ymax > 1e10:
          ymax = 10
        
        self.g('set autoscale x')
        self.g('set yrange ['+str(ymin)+':'+str(ymax)+']')
        self.g('set xlabel "Time [s]"')
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
    def __init__(self, g, dm, in_nsamps):
        self.g = g
        self.dm = dm
        self.in_nsamps = in_nsamps
    def plot(self, data):
        #self.g.reset()
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

        tsamp = 0.000064
        in_ntime = tsamp * in_nsamps
        t0 = 0 - (in_ntime / 2.0)
        t1 = 0 + (in_ntime / 2.0)
        self.g('t0 = ' + str(t0) + "\n")
        self.g('t1 = ' + str(t1) + "\n")
        self.g('f0 = 1581.')
        self.g('f1 = 1181.')
        self.g('dm = '+str(dm))
        self.g('k = 4.148808e3')
        self.g('g(x) = sqrt(1. / ((x-'+str(in_ntime/8.0)+') / (dm*k) + 1./f0**2))')

        self.g('set x2range [t0:t1]')
        self.g('set y2range [f0:f1]')

        # TODO: Work out equation for DM curve in pixel coords
        self.g('plot "' + data + '" matrix with image notitle, g(x) w l notitle lw 2 lc 2 axes x2y2')

def plotCandDspsr(fil_file, utc_start, sample, filter, dm):

  Dada.logMsg(1, DL, "plotCandDspsr: utc_start =" + utc_start)
  ddd_time = convertTime(utc_start)
  Dada.logMsg(1, DL, "plotCandDspsr: ddd_time=" + ddd_time)

  # cmd = "getMJD " + ddd_time + " |& tail -n 1"
  # p = os.popen(cmd)
  # utc_start_mjd = p.readline().strip()
  # p.close()
  # Dada.logMsg(1, DL, "main: utc_start_mjd=" + utc_start_mjd )

  cand_time = (0.000064 * sample)
  cmd = "dmsmear -f 1382 -b 400 -n 1024 -d " + str(dm) + " -q 2>&1 "
  p = os.popen(cmd)
  cand_band_smear = p.readline().strip()
  p.close()
  Dada.logMsg(1, DL, "plotCandDspsr: cand_band_smear=" + str(float(cand_band_smear) * 1000) + " ms")

  cand_filter_time = (2 ** filter) * 0.000064
  Dada.logMsg(1, DL, "plotCandDspsr cand_filter_time=" + str(cand_filter_time * 1000) + " ms")

  cand_smearing = float(cand_band_smear) + float(cand_filter_time)

  Dada.logMsg(1, DL, "plotCandDspsr cand_smearing=" + str(cand_smearing * 1000) + " ms")

  cand_start_time = cand_time
  cand_tot_time   = cand_smearing

  cmd = "dspsr " + fil_file + " -S " + str(cand_start_time) + \
        " -b 512 " + \
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
  # cmd = "psrplot -j 'zap chan 0-160' -c y:win=1525:1182 -p freq ./" + archive + " -D -/PNG";
  cmd = "psrplot -J /home/dada/linux_64/bin/zap.psh -j 'zap chan 0-160,335-338' -c x:unit=ms -p freq+ ./" + archive + " -j 'F 128' -D -/PNG";
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

  # check the file exists
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


def convertTime (dada_time):
  time = datetime.datetime.strptime(dada_time, "%Y-%m-%d-%H:%M:%S")
  ddd_time = time.strftime("%Y-%j-%H:%M:%S")
  return ddd_time


############################################################################### 
#
# main
#

cfg = Bpsr.getConfig()
log_file = trans_paths.getControlDir() + "/" + LOGFILE
pid_file = trans_paths.getControlDir() + "/" + PIDFILE
quit_file = trans_paths.getControlDir() + "/"  + QUITFILE

if os.path.exists(quit_file):
  sys.stderr.write("quit file existed at launch: " + quit_file)
  sys.exit(1)

# become a daemon
# Dada.daemonize(pid_file, log_file)

try:

  Dada.logMsg(1, DL, "STARTING SCRIPT")

  quit_event = threading.Event()

  signal.signal(signal.SIGINT, signal_handler)

  # start a control thread to handle quit requests
  control_thread = Dada.controlThread(quit_file, pid_file, quit_event, DL)
  control_thread.start()

  Dada.logMsg(2, DL, "main")

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  Dada.logMsg(2, DL, "main: binding to " + LISTEN_HOST + ":" + str(LISTEN_PORT))
  sock.bind((LISTEN_HOST, LISTEN_PORT))

  # listen for at most 2 connections at a time (self)
  Dada.logMsg(2, DL, "main: sock.listen(2)")
  sock.listen(2)

  can_read = [sock]
  can_write = []
  can_error = []
  timeout = 1

  # keep listening
  while (not quit_event.isSet()):

    Dada.logMsg(3, DL, "main: calling select len(can_read)="+str(len(can_read)))
    timeout = 1
    did_read, did_write, did_error = select.select(can_read, can_write, can_error, timeout)
    Dada.logMsg(3, DL, "main: read="+str(len(did_read))+" write="+str(len(did_write))+" error="+str(len(did_error)))  

    # if we did_read
    if (len(did_read) > 0):
      for handle in did_read:
        if (handle == sock):
          (new_conn, addr) = sock.accept()
          Dada.logMsg(2, DL, "main: accept connection from "+repr(addr))
          # add the accepted connection to can_read
          can_read.append(new_conn)

        # an accepted connection must have generated some data
        else:
          message = handle.recv(4096)
          message = message.strip()
          Dada.logMsg(3, DL, "main: message='" + message+"'")

          if (len(message) == 0):
            Dada.logMsg(1, DL, "main: closing connection")
            handle.close()
            for i, x in enumerate(can_read):
              if (x == handle):
                del can_read[i]

          else:
            proc_type = "cand"
            Dada.logMsg(1, DL, "<- " + message)
            parts = message.split()
            for part in parts:
              Dada.logMsg(3, DL, "main: part=" + part)
              (key, val) = part.split('=')
              Dada.logMsg(4, DL, "main: key=" + key + " val=" + val)
              if key == "utc_start":
                utc_start = val
              elif key == "beam":
                beam = val
              elif key == "sample":
                sample = int(val)
              elif key == "filter":
                filter = int(val)
              elif key == "dm":
                dm = float(val)
              elif key == "proc_type":
                proc_type = val
              else:
                Dada.logMsg(1, DL, "main: unrecognized key/val pair: " + part)

            # find the .fil file
            cmd = "ls -1 " + FILTERBANK_DIR + "/*/" + utc_start + "/" + beam + "/" + utc_start + ".fil "
            Dada.logMsg(3, DL, "main: " + cmd)

            p = os.popen(cmd)
            fil_file = p.readline().strip()
            p.close()

            Dada.logMsg(1, DL, "main: proc_type="+proc_type)
      
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
                binary_data = plotCandDspsr(fil_file, utc_start, sample, filter, dm)
                binary_len = len(binary_data)
                Dada.logMsg(3, DL, "main: sending binary data len="+str(binary_len))
                handle.send(binary_data)

              else:
                Dada.logMsg(2, DL, "main: plotCandidate()")
                binary_data = plotCandidate(fil_file, sample, filter, dm)
                binary_len = len(binary_data)
                Dada.logMsg(3, DL, "main: sending binary data len="+str(binary_len))
                handle.send(binary_data)

            Dada.logMsg(2, DL, "main: closing handle")
            handle.close()
            for i, x in enumerate(can_read):
              if (x == handle):
                del can_read[i]


except:
  Dada.logMsg(-2, DL, "main: exception caught: " + str(sys.exc_info()[0]))
  print '-'*60
  traceback.print_exc(file=sys.stdout)
  print '-'*60

quit_event.set()


for i, handle in enumerate(can_read):
  Dada.logMsg(2, DL, "main: closing can_read["+str(i)+"]")
  handle.close
  del can_read[i]

if (not sock == []): 
  Dada.logMsg(2, DL, "main: closing server socket [2]")
  sock.close()

Dada.logMsg(2, DL, "main: exiting")     

Dada.logMsg(2, DL, "main: joining control thread")
if (control_thread):
  control_thread.join()

Dada.logMsg(1, DL, "STOPPING SCRIPT")

# exit
sys.exit(0)

