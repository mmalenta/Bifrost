#!/usr/bin/env python 

import sys
import numpy as np
from math import sin, pi

class Classifier(object):
    def __init__(self, gb):
        self.sin_gb       = sin((pi/180)*gb)
        self.egal_cut     = 100
        self.max_nbeams   = 3
        self.members_cut  = 3
        self.filter_cut   = 0
        self.snr_cut      = 0
   
    # test if candidate is galactic
    def is_galactic(self, cand):
        return (cand['dm'] * self.sin_gb) <= self.egal_cut

    # test if candidate appears in multiple beams
    def is_coincident(self, cand):
        return cand['nbeams'] > self.max_nbeams

    # only allow candidates that are relatively short
    def is_too_long(self, cand):
        return cand['filter'] > self.filter_cut

    # only allow candidates above a SNR threshold
    def is_too_dim(self, cand):
        return cand['snr'] < self.snr_cut

    def is_noise(self, cand):
        return cand['members'] < self.members_cut


class TextOutput(object):
    def __init__(self):
        self.dm_base = 1.0
        self.snr_min = 6.0

    def print_html(self, data):
        if len(data['valid']) > 0:
            sys.stdout.write("<table width='100%' border=1 cellpadding=4px cellspacing=4px>\n")
            sys.stdout.write("<tr><th align=left>SNR</th><th align=left>Time</th><th align=left>DM</th><th align=left>Filter [ms]</th><th align=left>Beam</th></tr>\n")
            for (i, item) in enumerate(data['valid']['snr']):
                sys.stdout.write ("<tr>" + \
                                  "<td>" + str(data['valid']['snr'][i]) + "</td>" + \
                                  "<td>" + str(data['valid']['time'][i]) + "</td>" + \
                                  "<td>" + str(data['valid']['dm'][i]) + "</td>" + \
                                  "<td>" + str(0.064 * (2 **data['valid']['filter'][i])) + "</td>" + \
                                  "<td>" + str(data['valid']['prim_beam'][i]) + "</td>" + \
                                  "</tr>\n")
            sys.stdout.write("</table>\n")

    def print_text(self, data):
        if len(data['valid']) > 0:
            for (i, item) in enumerate(data['valid']['snr']):
                sys.stdout.write (str(data['valid']['snr'][i]) + "\t" + \
                                  str(data['valid']['time'][i]) + "\t" + \
                                  str(data['valid']['samp_idx'][i]) + "\t" + \
                                  str(data['valid']['dm'][i]) + "\t" + \
                                  str(data['valid']['filter'][i]) + "\t" + \
                                  str(data['valid']['prim_beam'][i]) + "\t" + \
                                  "\n")
    def print_xml(self, data):
        # get indicie list for sorting via snr
        snr_sorted_indices = [i[0] for i in sorted(enumerate(data['valid']['snr']), key=lambda x:x[1],reverse=True)]

        cand_i = 0
        for i in snr_sorted_indices:
            cand_i += 1
            sys.stdout.write ("<candidate snr='" + str(data['valid']['snr'][i]) + \
                                       "' time='" + str(data['valid']['time'][i]) + \
                                       "' dm='" + str(data['valid']['dm'][i]) + \
                                       "' samp_idx='" + str(data['valid']['samp_idx'][i]) + \
                                       "' filter='" + str(data['valid']['filter'][i]) + \
                                       "' prim_beam='" + str(data['valid']['prim_beam'][i] + 1) + "'/>\n")




if __name__ == "__main__":
    import argparse
    import Gnuplot
    
    parser = argparse.ArgumentParser(description="Detects FRB's in candidates file")
    parser.add_argument('-cands_file', default="all_candidates.dat")
    parser.add_argument('-snr_cut', type=float, default=10)
    parser.add_argument('-gb', type=float)
    parser.add_argument('-filter_cut', type=int, default=8)
    parser.add_argument('-cand_list_xml', action="store_true")
    parser.add_argument('-cand_list_html', action="store_true")
    parser.add_argument('-verbose', action="store_true")
    args = parser.parse_args()
    
    filename = args.cands_file
    verbose = args.verbose
    cand_list_xml = args.cand_list_xml
    cand_list_html = args.cand_list_html
    
    # Load candidates from all_candidates file
    all_cands = \
        np.loadtxt(filename,
                   dtype={'names': ('snr','samp_idx','time','filter',
                                    'dm_trial','dm','members','begin','end',
                                    'nbeams','beam_mask','prim_beam',
                                    'max_snr','beam'),
                          'formats': ('f4', 'i4', 'f4', 'i4',
                                      'i4', 'f4', 'i4', 'i4', 'i4',
                                      'i4', 'i4', 'i4',
                                      'f4', 'i4')})
    if verbose:
      sys.stderr.write ("Loaded %i candidates\n" % len(all_cands))

    
    classifier = Classifier(args.gb)
    classifier.snr_cut = args.snr_cut
    classifier.filter_cut = args.filter_cut
    
    # Filter candidates based on classifications
    if verbose:
      sys.stderr.write ("Classifying candidates...\n")

    categories = {}
    is_coincident = classifier.is_coincident(all_cands)
    is_galactic   = (is_coincident == False) & classifier.is_galactic(all_cands)
    is_too_long   = (is_coincident == False) & (is_galactic == False) & classifier.is_too_long(all_cands)
    is_too_dim    = (is_coincident == False) & (is_galactic == False) & (is_too_long == False) & classifier.is_too_dim(all_cands)
    is_noise      = (is_coincident == False) & (is_galactic == False) & (is_too_long == False) & (is_too_dim == False) & classifier.is_noise(all_cands)
    is_valid      = (is_coincident == False) & (is_galactic == False) & (is_too_long == False) & (is_too_dim == False) & (is_noise == False)

    categories["coincident"] = all_cands[is_coincident]
    categories["galactic"]   = all_cands[is_galactic]
    categories["too_long"]   = all_cands[is_too_long]
    categories["too_dim"]    = all_cands[is_too_dim]
    categories["valid"]      = all_cands[is_valid]
    
    if verbose:
      sys.stderr.write ( "Classified %i as coincident\n" % len(categories["coincident"]))
      sys.stderr.write ( "           %i as galactic spikes\n" % len(categories["galactic"]))
      sys.stderr.write ( "           %i as fat events\n" % len(categories["too_long"]))
      sys.stderr.write ( "           %i as dim events\n" % len(categories["too_dim"]))
      sys.stderr.write ( "           %i as valid FRB candidates\n" % len(categories["valid"]))

    text_output = TextOutput()

    if cand_list_xml:
      text_output.print_xml(categories)
    elif cand_list_html:
      text_output.print_html(categories)
    else:
      text_output.print_text(categories)

    if verbose:
      sys.stderr.write ( "Done\n")
