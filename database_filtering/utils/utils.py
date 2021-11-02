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

def generate_bonds(mol):
    bonds = [(m.GetBeginAtomIdx(), m.GetEndAtomIdx()) for m in mol.GetBonds()]
    return bonds

def extract_core_and_r_group(init_mol, graph, linker):
    for atom in init_mol.GetAtoms():
        if atom.GetPDBResidueInfo().GetName().strip() == linker[0].strip():
            idx_linker_r_group = atom.GetIdx()
        elif atom.GetPDBResidueInfo().GetName().strip() == linker[1].strip():
            idx_linker_core = atom.GetIdx()
    graph.remove_node(idx_linker_r_group)
    components = nx.connected_components(graph)
    for component in components:
        if idx_linker_core in component:
            edit = Chem.EditableMol(init_mol)
            atomsToRemove = [i for i in component]
            atomsToRemove.sort(reverse=True)
            for atom in atomsToRemove:
                edit.RemoveAtom(atom)
            substructure = edit.GetMol()
            try:
                Chem.SanitizeMol(r_group)
            except:
                pass
        else:
            edit = Chem.EditableMol(init_mol)
            atomsToRemove = [i for i in component]
            atomsToRemove.append(idx_linker_r_group)
            atomsToRemove.sort(reverse=True)
            for atom in atomsToRemove:
                edit.RemoveAtom(atom)
            r_group = edit.GetMol()
            try:
                Chem.SanitizeMol(substructure)
            except:
                pass
    return r_group, substructure, idx_linker_core, idx_linker_r_group

def generate_graph(bonds):
    """
    For a given list of bonds, represented as relationships between atom indices, generate a graph of a molecule.

    Parameters
    ----------
    bonds: List of tuples. Each tuple consists of two atom indices.
    Returns
    -------
    G: Graph of the molecule.
    """
    G = nx.Graph()
    for i in bonds:
        G.add_edge(i[0], i[1])
    return G

def retrieve_r_groups(init_mol, substructure, r_group_idx):
    mol={}
    mol['core'] = []
    mol['Variable R_group'] = []
    mol['Dumped R_groups'] = []
    for atom in init_mol.GetAtoms():
        if atom.GetIdx() in substructure:
            mol['core'].append(atom.GetIdx())
        elif atom.GetIdx() in r_group_idx:
            mol['Variable R_group'].append(atom.GetIdx())
        else:
            mol['Dumped R_groups'].append(atom.GetIdx())
    return mol

def ligand_r_groups(df_init,matching,linker, idx_core, idx_r_group, ligand, init):
    allowed_r_groups = []
    actual_r_groups = []
    substructure_ligand = []
    check = []
    for pair in matching:
        if pair[0] == idx_r_group:
            allowed_r_groups.append(pair[1])
        substructure_ligand.append(pair[1])
    for bond in ligand.GetBonds():
        if bond.GetBeginAtomIdx() in substructure_ligand:
            if bond.GetEndAtomIdx() not in substructure_ligand:
                actual_r_groups.append(bond.GetBeginAtomIdx())
        elif bond.GetBeginAtomIdx() not in substructure_ligand:
            if bond.GetEndAtomIdx() in substructure_ligand:
                actual_r_groups.append(bond.GetEndAtomIdx())
    for idx in actual_r_groups:
        if idx in allowed_r_groups:
            check.append(1)
        else:
            check.append(0)
    if 0 in check:
        return False
    else:
        return True

def get_r_group_idx(init_mol, r_group):
    r_group_idx = []
    for atom_core in init_mol.GetAtoms():
        for atom_r in r_group.GetAtoms():
            if atom_core.GetPDBResidueInfo().GetName().strip() == atom_r.GetPDBResidueInfo().GetName().strip():
                r_group_idx.append(atom_core.GetIdx())
    return r_group_idx

def filter_mols(init_mol, ligands, outfile, linker):
    ligs = Chem.SDMolSupplier(ligands)
    init_mol = Chem.MolFromPDBFile(init_mol, removeHs=True)
    bonds = generate_bonds(init_mol)
    output = []
    for mol in ligs:
        copy = Chem.EditableMol(mol)
        graph = generate_graph(bonds)
        r_group, core, idx_linker_core, idx_linker_r_group = self.extract_core_and_r_group(init_mol, graph, linker)
        matching = generate_matching(core,mol)
        substructure = mol.GetSubstructMatches(core)
        if len(substructure) > 2:
            print('More than one substructure match for ligand %s, skipping.' % mol.GetProp("_Name"))
        if matching:
            print('Analyzing Ligand %s' % mol.GetProp("_Name"))
            core_idx = [atom.GetIdx() for atom in core.GetAtoms()]
            r_group_idx = get_r_group_idx(init_mol,r_group)
            df = retrieve_r_groups(init_mol,core_idx,r_group_idx)
            filtering = ligand_r_groups(df,matching,linker, idx_linker_core, idx_linker_r_group, copy.GetMol(), init_mol)
            if filtering:
                mol = copy.GetMol()
                output.append(mol)
        else:
            print('No substructure match for ligand %s, skipping' % mol.GetProp("_Name"))
    save_results(outfile,output)

def save_results(out, output):
    with Chem.SDWriter('%s.sdf' % out) as writer:
        for mol in output:
            try:
                Chem.SanitizeMol(mol)
                #print(Chem.MolToPDBBlock(mol))
                writer.write(mol)
            except:
                print('Save failed for %s' % mol.GetProp("_Name"))


