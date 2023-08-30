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
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import Descriptors, QED, RDConfig, AllChem, rdFMCS, rdMolAlign, TemplateAlign


def generate_matching(init_mol, ligand):
    substructure = ligand.GetSubstructMatches(init_mol)
    if substructure:
        try:
            match = rdDepictor.GenerateDepictionMatching2DStructure(ligand, init_mol)
        except:
            match = None
    else:
        print('ERROR: Ligand does not have substructure match with init mol.')
        match = None
    return match


def get_allowed_r_groups(init_mol, ligand, linkers):
    matching = generate_matching(init_mol, ligand)
    idx_atoms_core_ligand = []
    idx_ligand = []
    for linker in linkers:
        if matching:
            for pair in matching:
                idx_atoms_core_ligand.append(pair[1])
                if init_mol.GetAtomWithIdx(pair[0]).GetPDBResidueInfo().GetName().strip() == linker.strip():
                    idx_ligand.append(pair[1])
            else:
                return None, None
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
    #init_mol = Chem.MolFromPDBFile(init_mol, removeHs=True)
    from rdkit.Chem import rdFMCS

    output = []
    for file in ligands:
        ligs = Chem.SDMolSupplier(file)
        for mol in ligs:
            if mol:
                copy = Chem.EditableMol(mol)
            else:
                print('Failed to load molecule')
                continue
            params = rdFMCS.MCSParameters()
            params.BondTyper = rdFMCS.BondCompare.CompareAny
            compare = [init_mol, mol]
            res = rdFMCS.FindMCS(compare, params)
            res_mol = Chem.MolFromSmarts(res.smartsString)
            if len(res_mol.GetAtoms()) == len(init_mol.GetAtoms()) and len(res_mol.GetBonds()) == len(init_mol.GetBonds()):
                init_mol = res_mol
            else:
                print('No substructure match for molecule %s, skipping' % mol.GetProp("_Name"))
                continue
            substructure = mol.GetSubstructMatches(res_mol)
            if len(substructure) > 2:
                print('More than one substructure match for molecule %s, skipping.' % mol.GetProp("_Name"))
                continue
            if substructure:
                ligand_idx_allowed, idx_core_ligand = get_allowed_r_groups(init_mol,mol,linker)
                if ligand_idx_allowed is not None and idx_core_ligand is not None:
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

def generate_combinations(dictionary):
    from itertools import product

    values = list(dictionary.values())
    combinations = []

    for combination in product(*values):
        is_valid = True

        for i in range(len(combination) - 1):
            if combination[i] in values[i + 1]:
                is_valid = False
                break
        if is_valid:
            combinations.append(combination)

    return combinations

def generate_alternative_scaffolds(template_lig, scaffold_changes, linker):
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFMCS
    from itertools import combinations, chain
    import subprocess
    scaffold_changes_dict = {}
    combinations_mols = []
    linkers=[]
    seen_at_types={}
    atomic_nums = {
        'H': [1, 1],
        'He': [2, 0],
        'Li': [3, 1],
        'Be': [4, 2],
        'B': [5, 3],
        'C': [6, 4],
        'N': [7, 3],
        'O': [8, 2],
        'F': [9, 1],
        'Ne': [10, 0],
        'Na': [11, 1],
        'Mg': [12, 2],
        'Al': [13, 3],
        'Si': [14, 4],
        'P': [15, 3],
        'S': [16, 2],
        'Cl': [17, 1],
        'K': [19, 1],
        'Ar': [18, 0],
        'Ca': [20, 2],
        'Sc': [21, 3],
        'Ti': [22, 4],
        'V': [23, 5],
        'Cr': [24, 6],
        'Mn': [25, 7],
        'Fe': [26, 2],
        'Co': [27, 2],
        'Ni': [28, 2],
        'Cu': [29, 2],
        'Zn': [30, 2],
        'Ga': [31, 3],
        'Ge': [32, 4],
        'As': [33, 3],
        'Se': [34, 2],
        'Br': [35, 1],
        'Kr': [36, 0],
        'Rb': [37, 1],
        'Sr': [38, 2],
        'Y': [39, 3],
        'Zr': [40, 4],
        'Nb': [41, 5],
        'Mo': [42, 6],
        'Tc': [43, 7],
        'Ru': [44, 8],
        'Rh': [45, 9],
        'Pd': [46, 10],
        'Ag': [47, 1],
        'Cd': [48, 2],
        'In': [49, 3],
        'Sn': [50, 4],
        'Sb': [51, 3],
        'Te': [52, 2],
        'I': [53, 1],
        'Xe': [54, 0],
        'Cs': [55, 1],
        'Ba': [56, 2],
        'La': [57, 3],
        'Ce': [58, 4],
        'Pr': [59, 3],
        'Nd': [60, 3],
        'Pm': [61, 3],
        'Sm': [62, 3],
        'Eu': [63, 3],
        'Gd': [64, 3],
        'Tb': [65, 3],
        'Dy': [66, 3],
        'Ho': [67, 3],
        'Er': [68, 3],
        'Tm': [69, 3],
        'Yb': [70, 3],
        'Lu': [71, 3],
        'Hf': [72, 4],
        'Ta': [73, 5],
        'W': [74, 6],
        'Re': [75, 7],
        'Os': [76, 8],
        'Ir': [77, 9],
        'Pt': [78, 10],
        'Au': [79, 1],
        'Hg': [80, 2],
        'Tl': [81, 3],
        'Pb': [82, 4],
        'Bi': [83, 3],
        'Th': [90, 4],
        'Pa': [91, 5],
        'U': [92, 6],
        'Np': [93, 6],
        'Pu': [94, 6],
        'Am': [95, 6],
        'Cm': [96, 6],
        'Bk': [97, 6],
        'Cf': [98, 6],
        'Es': [99, 6],
        'Fm': [100, 6],
        'Md': [101, 6],
        'No': [102, 6],
        'Lr': [103, 3]
    }
    atom_dict = {
        'H': 1,
        'He': 2,
        'Li': 3,
        'Be': 4,
        'B': 5,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'Ne': 10,
        'Na': 11,
        'Mg': 12,
        'Al': 13,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17,
        'K': 19,
        'Ar': 18,
        'Ca': 20,
        'Sc': 21,
        'Ti': 22,
        'V': 23,
        'Cr': 24,
        'Mn': 25,
        'Fe': 26,
        'Ni': 28,
        'Co': 27,
        'Cu': 29,
        'Zn': 30,
        'Ga': 31,
        'Ge': 32,
        'As': 33,
        'Se': 34,
        'Br': 35,
        'Kr': 36,
        'Rb': 37,
        'Sr': 38,
        'Y': 39,
        'Zr': 40,
        'Nb': 41,
        'Mo': 42,
        'Tc': 43,
        'Ru': 44,
        'Rh': 45,
        'Pd': 46,
        'Ag': 47,
        'Cd': 48,
        'In': 49,
        'Sn': 50,
        'Sb': 51,
        'Te': 52,
        'I': 53,
        'Xe': 54,
        'Cs': 55,
        'Ba': 56,
        'La': 57,
        'Ce': 58,
        'Pr': 59,
        'Nd': 60,
        'Pm': 61,
        'Sm': 62,
        'Eu': 63,
        'Gd': 64,
        'Tb': 65,
        'Dy': 66,
        'Ho': 67,
        'Er': 68,
        'Tm': 69,
        'Yb': 70,
        'Lu': 71,
        'Hf': 72,
        'Ta': 73,
        'W': 74,
        'Re': 75,
        'Os': 76,
        'Ir': 77,
        'Pt': 78,
        'Au': 79,
        'Hg': 80,
        'Tl': 81,
        'Pb': 82,
        'Bi': 83,
        'Th': 90,
        'Pa': 91,
        'U': 92,
        'Np': 93,
        'Pu': 94,
        'Am': 95,
        'Cm': 96,
        'Bk': 97,
        'Cf': 98,
        'Es': 99,
        'Fm': 100,
        'Md': 101,
        'No': 102,
        'Lr': 103,
    }
    rev_atom_nums = {v: k for k, v in atom_dict.items()}

    for ref_atom_group in scaffold_changes.split('-'):
        ref_atom = ref_atom_group.split(':')[0]
        atom_pos_group = ref_atom_group.split(':')[1]
        scaffold_changes_dict[ref_atom] = []
        for at_pos in atom_pos_group.split(','):
            scaffold_changes_dict[ref_atom].append(f"{ref_atom},{at_pos}")

    combinations = generate_combinations(scaffold_changes_dict)
    mol = Chem.MolFromPDBFile(template_lig, removeHs=True, sanitize=True)

    for i,combination in enumerate(combinations):
        init_mol = Chem.EditableMol(mol)


        idxs = get_idxs(combination,mol)

        for at in mol.GetAtoms():
            for bond in at.GetNeighbors():
                init_mol.RemoveBond(at.GetIdx(), bond.GetIdx())
                init_mol.AddBond(at.GetIdx(), bond.GetIdx(), Chem.rdchem.BondType.SINGLE)

            ref_at_type = at.GetPDBResidueInfo().GetName().strip()[0]
            if ref_at_type not in seen_at_types:
                seen_at_types[ref_at_type] = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == at.GetAtomicNum()])

        init_mol = init_mol.GetMol()
        for (type,idx) in idxs:

            if init_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().strip()[0] == type:
                init_mol.GetAtomWithIdx(idx).SetFormalCharge(0)
                continue

            init_mol.GetAtomWithIdx(idx).SetAtomicNum(atomic_nums[type][0])
            init_mol.GetAtomWithIdx(idx).SetFormalCharge(0)
            new_res_name = f"{type}{str(seen_at_types[type] + 1)}"
            Chem.MolToPDBFile(init_mol, f"res_{i}.pdb")            #init_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().SetName(init_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().replace(init_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().strip(), new_res_name))
            init_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().SetName(new_res_name)
            seen_at_types[type] = seen_at_types[type] + 1
            if init_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().strip() in linker:
                index_linker = linker.index(init_mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().strip())
                linker[index_linker] = new_res_name
        #pdb_res_name = at.GetPDBResidueInfo().GetName().strip()
        #print(pdb_res_name, combination_at.split(',')[0])
        #if pdb_res_name == combination_at.split(',')[0]:
            #if atomic_nums[at_type][0] == at.GetAtomicNum():
            #    continue
        #    breakpoint()
        #    at.SetAtomicNum(atomic_nums[at_type][0])
        #    at.SetFormalCharge(1)
        #    new_res_name=f"{at_type}{str(seen_at_types[at_type]+1)}"
        #    at.GetPDBResidueInfo().SetName(at.GetPDBResidueInfo().GetName().replace(pdb_res_name,new_res_name))
        #    seen_at_types[at_type] = seen_at_types[at_type] +1
        #    if pdb_res_name in linker:
        #        index_linker = linker.index(pdb_res_name)
        #        linker[index_linker] = new_res_name
        #else:
        #    continue

        #init_mol = fix_valences(init_mol, atomic_nums)

        rdDetermineBonds.DetermineConnectivity(init_mol, charge=0)
        #init_mol = Chem.AddHs(init_mol, addCoords=True)
        Chem.MolToPDBFile(init_mol,f"res_{i}.pdb")
        print(i,Chem.MolToSmiles(init_mol))
        #add_nitrogen_charges(init_mol)
        #add_oxygen_charges(init_mol)
        #try:
            #AllChem.EmbedMolecule(init_mol)
        #except:
        #    print(f"Failed for combination {combination}")
        #    continue
        #    continue
        #crdDetermineBonds.DetermineBondOrders(init_mol, charge=0)
        #AllChem.Compute2DCoords(Chem.MolToSmiles(init_mol))
        #init_mol = Chem.rdmolops.RemoveAllHs(init_mol)
        combinations_mols.append(init_mol)
        linkers.append(linker)
        #subprocess.run(f"sbatch run_rgroups_s.sh '{Chem.MolToSmiles(init_mol)}' {linker[0]} {linker[1]}", shell=True)


    import sys
    sys.exit()
    return combinations_mols, linkers

def get_idxs(combination, mol):
    import rdkit
    from rdkit import Chem
    idxs = []
    for combination_at in combination:
        for at in mol.GetAtoms():
            if at.GetPDBResidueInfo().GetName().strip() == combination_at.split(',')[0]:
                idxs.append((combination_at.split(',')[1],at.GetIdx()))
            else:
                continue
    return idxs

def fix_valences(mol,atomic_nums):
    atom_dict = {
        'H': 1,
        'He': 2,
        'Li': 3,
        'Be': 4,
        'B': 5,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
        'Ne': 10,
        'Na': 11,
        'Mg': 12,
        'Al': 13,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17,
        'K': 19,
        'Ar': 18,
        'Ca': 20,
        'Sc': 21,
        'Ti': 22,
        'V': 23,
        'Cr': 24,
        'Mn': 25,
        'Fe': 26,
        'Ni': 28,
        'Co': 27,
        'Cu': 29,
        'Zn': 30,
        'Ga': 31,
        'Ge': 32,
        'As': 33,
        'Se': 34,
        'Br': 35,
        'Kr': 36,
        'Rb': 37,
        'Sr': 38,
        'Y': 39,
        'Zr': 40,
        'Nb': 41,
        'Mo': 42,
        'Tc': 43,
        'Ru': 44,
        'Rh': 45,
        'Pd': 46,
        'Ag': 47,
        'Cd': 48,
        'In': 49,
        'Sn': 50,
        'Sb': 51,
        'Te': 52,
        'I': 53,
        'Xe': 54,
        'Cs': 55,
        'Ba': 56,
        'La': 57,
        'Ce': 58,
        'Pr': 59,
        'Nd': 60,
        'Pm': 61,
        'Sm': 62,
        'Eu': 63,
        'Gd': 64,
        'Tb': 65,
        'Dy': 66,
        'Ho': 67,
        'Er': 68,
        'Tm': 69,
        'Yb': 70,
        'Lu': 71,
        'Hf': 72,
        'Ta': 73,
        'W': 74,
        'Re': 75,
        'Os': 76,
        'Ir': 77,
        'Pt': 78,
        'Au': 79,
        'Hg': 80,
        'Tl': 81,
        'Pb': 82,
        'Bi': 83,
        'Th': 90,
        'Pa': 91,
        'U': 92,
        'Np': 93,
        'Pu': 94,
        'Am': 95,
        'Cm': 96,
        'Bk': 97,
        'Cf': 98,
        'Es': 99,
        'Fm': 100,
        'Md': 101,
        'No': 102,
        'Lr': 103,
    }
    rev_atom_nums = {v: k for k, v in atom_dict.items()}

    valences = {}
    emol = Chem.EditableMol(mol)
    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType())
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        emol.RemoveBond(atom1, atom2)
        at1 = mol.GetAtomWithIdx(atom1)
        at2 = mol.GetAtomWithIdx(atom2)
        emol.AddBond(atom1, atom2, Chem.rdchem.BondType.SINGLE)


    m = emol.GetMol()
    return m

def add_nitrogen_charges(m):
    import rdkit
    from rdkit import Chem
    m.UpdatePropertyCache(strict=False)
    #ps = Chem.DetectChemistryProblems(m)
    #if not ps:
    #    Chem.SanitizeMol(m)
    #    return m
    for at in m.GetAtoms():
        if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
            at.SetFormalCharge(1)
    return m

def add_oxygen_charges(m):
    import rdkit
    from rdkit import Chem
    m.UpdatePropertyCache(strict=False)
    for at in m.GetAtoms():
        if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
            at.SetFormalCharge(1)
    Chem.SanitizeMol(m)
    return m



