"""
PDBBind binding pocket dataset loader.
"""

import os
import numpy as np
import pandas as pd
import time
import re
from rdkit import Chem
import deepchem as dc

def compute_binding_pocket_features(pocket_featurizer, ligand_featurizer,
                                    pdb_subdir, pdb_code, threshold=.3):
  """Compute features for a given complex"""
  protein_file = os.path.join(pdb_subdir, "%s_protein.pdb" % pdb_code)
  ligand_file = os.path.join(pdb_subdir, "%s_ligand.sdf" % pdb_code)
  ligand_mol2 = os.path.join(pdb_subdir, "%s_ligand.mol2" % pdb_code)

  # Extract active site
  active_site_box, active_site_atoms = (
      dc.dock.binding_pocket.extract_active_site(
          protein_file, ligand_file))

  # Featurize ligand
  mol = Chem.MolFromMol2File(str(ligand_mol2), removeHs=False)
  if mol is None:
    return None, None
  # Default for CircularFingerprint
  n_ligand_features = 1024
  ligand_features = ligand_featurizer.featurize([mol])

  # Featurize pocket
  finder = dc.dock.ConvexHullPocketFinder()
  pockets = finder.find_pockets(protein_file)
  n_pockets = len(pockets)
  n_pocket_features = dc.feat.BindingPocketFeaturizer.n_features

  features = np.zeros((n_pockets, n_pocket_features+n_ligand_features))
  pocket_features = pocket_featurizer.featurize(
      protein_file, pockets)
  # Note broadcast operation
  features[:, :n_pocket_features] = pocket_features
  features[:, n_pocket_features:] = ligand_features

  # Compute labels for pockets
  labels = np.zeros(n_pockets)
  pocket_atoms[active_site_box] = active_site_atoms
  for ind, pocket in enumerate(pockets):
    overlap = dc.dock.binding_pocket.compute_overlap(
        pocket_atoms, active_site_box, pocket)
    if overlap > threshold:
      labels[ind] = 1
    else:
      labels[ind] = 0 
  return features, labels

def load_pdbbind_labels(labels_file):
  """Loads pdbbind labels as dataframe"""
  # Some complexes have labels but no PDB files. Filter these manually
  missing_pdbs = ["1d2v", "1jou", "1s8j", "1cam", "4mlt", "4o7d"]
  contents = []
  with open(labels_file) as f:
    for line in f:
      if line.startswith("#"):
        continue
      else:
        # Some of the ligand-names are of form (FMN ox). Use regex
        # to merge into form (FMN-ox)
        p = re.compile('\(([^\)\s]*) ([^\)\s]*)\)')
        line = p.sub('(\\1-\\2)', line)
        elts = line.split()
        # Filter if missing PDB files
        if elts[0] in missing_pdbs:
          continue
        contents.append(elts)
  contents_df = pd.DataFrame(
      contents,
      columns=("PDB code", "resolution", "release year", "-logKd/Ki", "Kd/Ki",
               "ignore-this-field", "reference", "ligand name"))
  return contents_df

def featurize_pdbbind_pockets(data_dir=None, subset="core"):
  """Featurizes pdbbind according to provided featurization"""
  tasks = ["active-site"]
  current_dir = os.path.dirname(os.path.realpath(__file__))
  data_dir = os.path.join(current_dir, "%s_pockets" % (subset))
  if os.path.exists(data_dir):
    return dc.data.DiskDataset(data_dir), tasks
  pdbbind_dir = os.path.join(current_dir, "../../v2015")

  # Load PDBBind dataset
  if subset == "core":
    labels_file = os.path.join(pdbbind_dir, "INDEX_core_data.2013")
  elif subset == "refined":
    labels_file = os.path.join(pdbbind_dir, "INDEX_refined_data.2015")
  elif subset == "full":
    labels_file = os.path.join(pdbbind_dir, "INDEX_general_PL_data.2015")
  else:
    raise ValueError("Only core, refined, and full subsets supported.")
  print("About to load contents.")
  if not os.path.exists(labels_file):
    raise ValueError("Run ../pdbbind/get_pdbbind.sh to download dataset, or check the path")
  contents_df = load_pdbbind_labels(labels_file)
  ids = contents_df["PDB code"].values
  y = np.array([float(val) for val in contents_df["-logKd/Ki"].values])

  # Define featurizers
  pocket_featurizer = dc.feat.BindingPocketFeaturizer()
  ligand_featurizer = dc.feat.CircularFingerprint(size=1024)

  # Featurize Dataset
  all_features = []
  all_labels = []
  missing_pdbs = []
  all_ids = []
  time1 = time.time()
  for ind, pdb_code in enumerate(ids):
    print("Processing complex %d, %s" % (ind, str(pdb_code)))
    pdb_subdir = os.path.join(pdbbind_dir, pdb_code)
    if not os.path.exists(pdb_subdir):
      print("%s is missing!" % pdb_subdir)
      missing_pdbs.append(pdb_subdir)
      continue
    features, labels = compute_binding_pocket_features(
        pocket_featurizer, ligand_featurizer, pdb_subdir, pdb_code)
    if features is None:
      print("Featurization failed!")
      continue
    all_features.append(features)
    all_labels.append(labels)
    ids = np.array(["%s%d" % (pdb_code, i) for i in range(len(labels))])
    all_ids.append(ids)
  time2 = time.time()
  print("TIMING: PDBBind Pocket Featurization took %0.3f s" % (time2-time1))
  X = np.vstack(all_features)
  y = np.concatenate(all_labels)
  w = np.ones_like(y)
  ids = np.concatenate(all_ids)
   
  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, data_dir=data_dir)
  return dataset, tasks

def load_pdbbind_pockets(split="index", subset="core"):
  """Load PDBBind datasets. Does not do train/test split"""
  dataset, tasks = featurize_pdbbind_pockets(subset=subset)

  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter()}
  splitter = splitters[split]
  ########################################################### DEBUG
  print("dataset.X.shape")
  print(dataset.X.shape)
  print("dataset.y.shape")
  print(dataset.y.shape)
  print("dataset.w.shape")
  print(dataset.w.shape)
  print("dataset.ids.shape")
  print(dataset.ids.shape)
  ########################################################### DEBUG
  train, valid, test = splitter.train_valid_test_split(dataset)

  transformers = []
  for transformer in transformers:
    train = transformer.transform(train)
  for transformer in transformers:
    valid = transformer.transform(valid)
  for transformer in transformers:
    test = transformer.transform(test)
  
  return tasks, (train, valid, test), transformers
