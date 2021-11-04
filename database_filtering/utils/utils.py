import gzip
import os
import rdkit
import networkx as nx
import shutil
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDepictor
from rdkit.Chem import PandasTools
from rdkit.Chem import rdRGroupDecomposition
from rdkit import RDLogger
from rdkit.Chem import Descriptors, QED, RDConfig, AllChem, rdFMCS, rdMolAlign, TemplateAlign



def generate_matching(init_mol, ligand):
    substructure = ligand.GetSubstructMatches(init_mol)
    if substructure:
        match = rdDepictor.GenerateDepictionMatching2DStructure(ligand, init_mol)
    else:
        print('ERROR: Ligand does not have substructure match with init mol.')
        match = None
    return match

def get_allowed_r_groups(init_mol, ligand, linkers):
    matching = generate_matching(init_mol, ligand)
    idx_atoms_core_ligand = []
    idx_ligand = []
    for linker in linkers:
        for pair in matching:
            idx_atoms_core_ligand.append(pair[1])
            if init_mol.GetAtomWithIdx(pair[0]).GetPDBResidueInfo().GetName().strip() == linker.strip():
                idx_ligand.append(pair[1])
    return idx_ligand, idx_atoms_core_ligand

def check_connections_to_core(ligand,ligand_idx_allowed, idx_core_ligand):
    filter = []
    for bond in ligand.GetBonds():
        if bond.GetBeginAtomIdx() in idx_core_ligand and bond.GetEndAtomIdx() not in idx_core_ligand:
            if bond.GetBeginAtomIdx() in ligand_idx_allowed:
                filter.append(1)
            else:
                filter.append(0)
        elif bond.GetBeginAtomIdx() not in idx_core_ligand and bond.GetEndAtomIdx() in idx_core_ligand:
            if bond.GetEndAtomIdx() in ligand_idx_allowed:
                filter.append(1)
            else:
                filter.append(0)
    if 0 in filter:
        return False
    else:
        return True


def filter_mols(init_mol, ligands, outfile, linker):
    init_mol = Chem.MolFromPDBFile(init_mol, removeHs=True)
    output = []
    for file in ligands:
        ligs = Chem.SDMolSupplier(file)
        for mol in ligs:
            if mol:
                copy = Chem.EditableMol(mol)
            else:
                print('Failed to load molecule')
                continue
            substructure = mol.GetSubstructMatches(init_mol)
            if len(substructure) > 2:
                print('More than one substructure match for molecule %s, skipping.' % mol.GetProp("_Name"))
                continue
            if substructure:
                ligand_idx_allowed, idx_core_ligand = get_allowed_r_groups(init_mol,mol,linker)
                if check_connections_to_core(mol, ligand_idx_allowed,idx_core_ligand):
                    mol = copy.GetMol()
                    output.append(mol)
                else:
                    print('Molecule %s did not meet the R-groups requirements.' % mol.GetProp("_Name"))
            else:
                print('No substructure match for molecule %s, skipping' % mol.GetProp("_Name"))
    save_results(outfile,output)

def save_results(out, output):
    with Chem.SDWriter('%s.sdf' % out) as writer:
        for mol in output:
            try:
                Chem.SanitizeMol(mol)
                writer.write(mol)
            except:
                print('Save failed for %s' % mol.GetProp("_Name"))


