# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:01:57 2020

@author: Zhenqin Wu
"""
import os
import numpy as np

def read_fasta(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  seq_descs = []
  seqs = []
  desc = None
  raw = ''
  for line in lines:
    if line[0] == '>':
      if desc != None:
        seq_descs.append(desc)
        seqs.append(raw)
        desc = None
        raw = ''
      desc = line.strip()[1:]
    else:
      raw += line.strip()
  if desc != None:
    seq_descs.append(desc)
    seqs.append(raw)
  return seq_descs, seqs


def write_fasta(sequence, path, name=None):
  if name is None:
    name = 'TEMP'
  with open(path, 'a') as f:
    f.writelines(['>%s\n' % name, '%s\n' % sequence, '\n'])
  return


def read_hhm(sequence, path, asterisks_replace=0.):
  with open(path, 'r') as f:
    hhm_file = f.read()
  profile_part = hhm_file.split('#')[-1]
  profile_part = profile_part.split('\n')
  whole_profile = [i.split() for i in profile_part]
  # This part strips away the header and the footer.
  whole_profile = whole_profile[5:-2]
  gap_profile = np.zeros((len(sequence), 10))
  aa_profile = np.zeros((len(sequence), 20))
  count_aa = 0
  count_gap = 0
  for line_values in whole_profile:
    if len(line_values) == 23:
      # The first and the last values in line_values are metadata, skip them.
      for j, t in enumerate(line_values[2:-1]):
        aa_profile[count_aa, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_aa += 1
    elif len(line_values) == 10:
      for j, t in enumerate(line_values):
        gap_profile[count_gap, j] = (
            2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
      count_gap += 1
    elif not line_values:
      pass
    else:
      raise ValueError('Wrong length of line %s hhm file. Expected 0, 10 or 23'
                       'got %d'%(line_values, len(line_values)))
  hmm_profile = np.hstack([aa_profile, gap_profile])
  return hmm_profile


def run_hhblits_local(sequence, path, name=None):
  if not os.path.exists(path, 'input.seq'):
    write_fasta(sequence, os.path.exists(path, 'input.seq'), name=name)
  
  commands = []
  commands.append('hhblits -v 1 -maxfilt 100000 -realign_max 100000 -all \
      -B 100000 -Z 100000 -diff inf -id 99 -cov 50 -i %s/input.seq -d \
      %s/UniRef30_2020_02 -oa3m %s/results.a3m -cpu 4 -n 3 -e 0.001' % \
      (path, os.environ['PNET_HHDB_PATH'], path))
  commands.append('reformat.pl -v 0 -r a3m clu %s/results.a3m %s/results.clu' \
      % (path, path))
  commands.append('reformat.pl -v 0 -r a3m fas %s/results.a3m %s/results.fas' \
      % (path, path))
  commands.append('hhmake -i %s/results.a3m' % path)
  return commands