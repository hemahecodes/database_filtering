import argparse as ap
import os
import glob
import pickle
import sys
import csv
import database_filtering
from database_filtering.utils.utils import filter_mols

def parse_args(args):
    parser = ap.ArgumentParser()
    parser.add_argument("-i","--template_ligand", type=str, required=True,
                        help="Path to PDB template ligand.")
    parser.add_argument("-l","--ligands", type=str, required=True,
                        help="Path to SDF file with ligands.")
    parser.add_argument("-o", "--outfile", required=True, help="Output file name.")
    parser.add_argument("-a", "--atom_linker", required = True, nargs='+', help = " Carbon atom of core that is bound to r-group.")
    parsed_args = parser.parse_args(args)
    return parsed_args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    template_lig = args.template_ligand
    ligands_path = args.ligands
    outfile= args.outfile
    linker = args.atom_linker
    dirs = glob.glob(ligands_path + "/*.sd*")
    ligands=[]
    i=0
    while i < len(dirs):
        if os.path.isdir(dirs[i]):
            dirs = dirs + glob.glob(dirs[i] + "/*")
        else:
            if os.path.splitext(dirs[i])[1] == '.sd' or os.path.splitext(dirs[i])[1] == ".sdf":
                ligands.append(dirs[i])
        i+=1
    filter_mols(template_lig,ligands,outfile,linker)
