from __future__ import print_function
import os
import pandas as pd
from rdkit import Chem
from glob import glob
import re
import joblib


def extract_labels(pdbbind_label_file):
    """Extract labels from pdbbind label file."""
    assert os.path.isfile(pdbbind_label_file)
    labels = {}
    with open(pdbbind_label_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                # lines in the label file have format
                # PDB-code Resolution Release-Year -logKd Kd reference ligand-name
                labels[parts[0]] = parts[3]
    return labels


def construct_df(pdb_stem_directory, pdbbind_label_file, pdbbind_df_joblib):
    """
    Constructs a DataFrame from pdb files and saves it using joblib.
    """
    labels = extract_labels(pdbbind_label_file)
    df_rows = []
    pdb_directories = [pdb for pdb in glob(os.path.join(pdb_stem_directory, '*/'))]

    for pdb_dir in pdb_directories:
        pdb_id = os.path.basename(os.path.normpath(pdb_dir))
        ligand_pdb = None
        protein_pdb = None
        ligand_mol2 = None

        for f in os.listdir(pdb_dir):
            if f.endswith("_ligand_hyd.pdb"):
                ligand_pdb = f
            elif f.endswith("_protein_hyd.pdb"):
                protein_pdb = f
            elif f.endswith("_ligand.mol2"):
                ligand_mol2 = f

        if not ligand_pdb or not protein_pdb or not ligand_mol2:
            print(f"Warning: Required files not present for {pdb_id}")
            continue

        ligand_pdb_path = os.path.join(pdb_dir, ligand_pdb)
        protein_pdb_path = os.path.join(pdb_dir, protein_pdb)
        ligand_mol2_path = os.path.join(pdb_dir, ligand_mol2)

        try:
            with open(protein_pdb_path, "r") as f:
                protein_pdb_lines = f.readlines()
            with open(ligand_pdb_path, "r") as f:
                ligand_pdb_lines = f.readlines()
            with open(ligand_mol2_path, "r") as f:
                ligand_mol2_lines = f.readlines()
        except IOError as e:
            print(f"Error reading file: {e}")
            continue

        print(f"About to compute ligand smiles string for {pdb_id}.")
        ligand_mol = Chem.MolFromPDBFile(ligand_pdb_path)
        if ligand_mol is None:
            print(f"Error: Could not convert ligand PDB to molecule for {pdb_id}.")
            continue
        smiles = Chem.MolToSmiles(ligand_mol)
        complex_id = f"{pdb_id}_{smiles}"
        label = labels.get(pdb_id, "Unknown")
        df_rows.append([pdb_id, smiles, complex_id, protein_pdb_lines,
                        ligand_pdb_lines, ligand_mol2_lines, label])

    pdbbind_df = pd.DataFrame(df_rows, columns=['pdb_id', 'smiles', 'complex_id',
                                                'protein_pdb', 'ligand_pdb',
                                                'ligand_mol2', 'label'])

    joblib.dump(pdbbind_df, pdbbind_df_joblib)
    print(f"DataFrame saved to {pdbbind_df_joblib}")
