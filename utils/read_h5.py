#!/usr/bin/env python3
import os
import h5py
import numpy as np
import argparse
from pprint import pprint

def print_attrs(name, obj):
    """Print attributes of an HDF5 object."""
    print(f"\nAttributes of {name}:")
    for key, val in obj.attrs.items():
        print(f"  {key}: {val}")

def explore_h5_file(h5_file, group=None, level=0):
    """Recursively explore and print contents of an HDF5 file."""
    if group is None:
        group = h5_file

    indent = "  " * level
    
    for key in group.keys():
        item = group[key]
        path = f"{group.name}/{key}" if group.name != "/" else f"/{key}"
        
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {path}")
            explore_h5_file(h5_file, item, level + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}Dataset: {path}, Shape: {item.shape}, Dtype: {item.dtype}")
            # Print sample data for small datasets
            if len(item.shape) == 0:
                print(f"{indent}  Value: {item[()]}")
            elif np.prod(item.shape) <= 10:
                print(f"{indent}  Values: {item[()]}")
            else:
                # For larger datasets, just print a few values
                if len(item.shape) == 1:
                    print(f"{indent}  First 5 values: {item[:5]}")
                else:
                    print(f"{indent}  Sample: {item[0]}")
            
            # Print dataset attributes
            if len(item.attrs) > 0:
                print(f"{indent}  Attributes:")
                for attr_name, attr_value in item.attrs.items():
                    print(f"{indent}    {attr_name}: {attr_value}")

def list_h5_files(directory):
    """List all H5 files in a directory."""
    h5_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            h5_files.append(os.path.join(directory, filename))
    return h5_files

def main():
    parser = argparse.ArgumentParser(description='Read and display contents of H5 files')
    parser.add_argument('--path', default='/home/mm/workbench/real/dataset/test/8', 
                        help='Path to directory containing H5 files')
    parser.add_argument('--file', help='Specific H5 file to read (optional)')
    args = parser.parse_args()
    
    if args.file:
        # Read a specific file
        file_path = os.path.join(args.path, args.file) if not os.path.isabs(args.file) else args.file
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return
        
        print(f"Reading H5 file: {file_path}")
        with h5py.File(file_path, 'r') as f:
            print("\nFile structure:")
            explore_h5_file(f)
    else:
        # List and read all H5 files in the directory
        if not os.path.exists(args.path):
            print(f"Error: Directory {args.path} does not exist")
            return
            
        h5_files = list_h5_files(args.path)
        
        if not h5_files:
            print(f"No H5 files found in {args.path}")
            return
            
        print(f"Found {len(h5_files)} H5 files in {args.path}")
        
        for i, file_path in enumerate(h5_files):
            print(f"\n\n===== File {i+1}/{len(h5_files)}: {os.path.basename(file_path)} =====")
            try:
                with h5py.File(file_path, 'r') as f:
                    print("\nFile structure:")
                    explore_h5_file(f)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

if __name__ == "__main__":
    main() 