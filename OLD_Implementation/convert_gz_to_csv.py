#!/usr/bin/env python3
"""
gzip_to_csv.py

Script to decompress gzipped CSV files back to regular CSV format.
Reads .csv.gz files from 'Results_Per_Section' directory and saves 
decompressed CSV files to 'Unzipped_Results_Per_Section' directory.

Author: Ahmed Yasser Hassanein
Date: 2025
"""

import os
import gzip
import shutil
import csv
import hashlib
from pathlib import Path
from datetime import datetime
import argparse

def verify_decompression_integrity(gzipped_path, csv_path):
    """
    Verify that a decompressed CSV file contains exactly the same data as the gzipped version
    Returns (success, message)
    """
    try:
        # Read original gzipped content
        with gzip.open(gzipped_path, 'rb') as f:
            gzipped_content = f.read()
        
        # Read decompressed CSV content
        with open(csv_path, 'rb') as f:
            csv_content = f.read()
        
        # Compare byte-by-byte
        if gzipped_content == csv_content:
            return True, "Content matches perfectly"
        else:
            return False, "Content mismatch detected"
            
    except Exception as e:
        return False, f"Error during verification: {str(e)}"

def verify_csv_readability(csv_path):
    """
    Verify that the decompressed CSV file can be read properly
    Returns (success, row_count, message)
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        return True, len(rows), f"CSV is readable with {len(rows)} rows"
        
    except Exception as e:
        return False, 0, f"CSV readability error: {str(e)}"

def decompress_gzip_to_csv(gzip_path, output_dir):
    """
    Decompress a single gzipped CSV file to regular CSV format
    Returns (success, csv_path, message)
    """
    gzip_path = Path(gzip_path)
    output_dir = Path(output_dir)
    
    # Determine output CSV filename
    if gzip_path.name.endswith('.csv.gz'):
        csv_filename = gzip_path.name[:-3]  # Remove .gz extension
    elif gzip_path.name.endswith('.gz'):
        csv_filename = gzip_path.name[:-3]  # Remove .gz extension
    else:
        csv_filename = gzip_path.stem + '.csv'
    
    csv_path = output_dir / csv_filename
    
    # Check if CSV already exists
    if csv_path.exists():
        return False, None, f"CSV file already exists: {csv_filename}"
    
    try:
        # Step 1: Decompress the file
        print(f"🔄 Decompressing: {gzip_path.name}")
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Step 2: Verify decompression integrity
        print(f"🔍 Verifying decompression integrity...")
        integrity_check, integrity_msg = verify_decompression_integrity(gzip_path, csv_path)
        if not integrity_check:
            csv_path.unlink()  # Delete failed CSV
            return False, None, f"Integrity verification failed: {integrity_msg}"
        
        # Step 3: Verify CSV readability
        print(f"🔍 Verifying CSV readability...")
        readable_check, row_count, readable_msg = verify_csv_readability(csv_path)
        if not readable_check:
            csv_path.unlink()  # Delete failed CSV
            return False, None, f"Readability verification failed: {readable_msg}"
        
        # Step 4: Compare file sizes
        gzipped_size = gzip_path.stat().st_size
        decompressed_size = csv_path.stat().st_size
        expansion_ratio = (decompressed_size / gzipped_size) if gzipped_size > 0 else 0
        
        print(f"📊 Expansion: {gzipped_size:,} → {decompressed_size:,} bytes ({expansion_ratio:.1f}x larger)")
        print(f"✅ Integrity check: {integrity_msg}")
        print(f"✅ Readability check: {readable_msg}")
        
        return True, csv_path, f"Successfully decompressed and verified ({row_count} rows)"
        
    except Exception as e:
        # Clean up on error
        if csv_path.exists():
            csv_path.unlink()
        return False, None, f"Decompression error: {str(e)}"

def find_gzipped_files(source_dir):
    """Find all gzipped files that could be CSV files"""
    source_dir = Path(source_dir)
    
    # Look for various gzip extensions
    patterns = ['*.csv.gz', '*.gz']
    gzip_files = []
    
    for pattern in patterns:
        gzip_files.extend(source_dir.glob(pattern))
    
    return sorted(set(gzip_files))  # Remove duplicates and sort

def process_gzipped_files(source_dir="Results_Per_Section", 
                         output_dir="Unzipped_Results_Per_Section", 
                         dry_run=False):
    """
    Process all gzipped files in the source directory
    
    Args:
        source_dir: Directory containing gzipped files
        output_dir: Directory to save decompressed CSV files
        dry_run: If True, only show what would be done without actually doing it
    """
    # Check if source directory exists
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Source directory '{source_dir}' not found!")
        print(f"📁 Current directory: {Path.cwd()}")
        available_dirs = [d.name for d in Path.cwd().iterdir() if d.is_dir()]
        print(f"📂 Available directories: {available_dirs}")
        return
    
    # Find all gzipped files
    gzip_files = find_gzipped_files(source_path)
    
    if not gzip_files:
        print(f"📁 No gzipped files found in '{source_dir}'")
        print(f"🔍 Looking for files with extensions: .csv.gz, .gz")
        return
    
    print(f"🎯 Found {len(gzip_files)} gzipped files in '{source_dir}'")
    print("-" * 60)
    
    # Show what we're going to process
    total_size = 0
    for gzip_file in gzip_files:
        size = gzip_file.stat().st_size
        total_size += size
        print(f"📦 {gzip_file.name} ({size:,} bytes)")
    
    print(f"\n💾 Total compressed size: {total_size:,} bytes ({total_size/(1024*1024):.2f} MB)")
    
    if dry_run:
        print(f"\n🔍 DRY RUN MODE - No files will be created")
        print(f"📂 Would create output directory: {output_dir}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📂 Output directory: {output_path.absolute()}")
    
    # Ask for confirmation
    response = input(f"\n❓ Proceed with decompressing {len(gzip_files)} files? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("❌ Operation cancelled")
        return
    
    # Process each file
    print(f"\n🚀 Starting decompression process...")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    total_compressed_size = 0
    total_decompressed_size = 0
    
    for i, gzip_file in enumerate(gzip_files, 1):
        print(f"\n[{i}/{len(gzip_files)}] Processing: {gzip_file.name}")
        print("-" * 40)
        
        compressed_size = gzip_file.stat().st_size
        total_compressed_size += compressed_size
        
        success, csv_path, message = decompress_gzip_to_csv(gzip_file, output_path)
        
        if success:
            decompressed_size = csv_path.stat().st_size
            total_decompressed_size += decompressed_size
            success_count += 1
            
            print(f"✅ SUCCESS: {gzip_file.name} → {csv_path.name}")
            print(f"📄 Result: {message}")
        else:
            error_count += 1
            print(f"❌ FAILED: {message}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 DECOMPRESSION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully decompressed: {success_count} files")
    print(f"❌ Failed: {error_count} files")
    
    if success_count > 0:
        expansion_ratio = total_decompressed_size / total_compressed_size if total_compressed_size > 0 else 0
        print(f"💾 Total size expansion: {total_compressed_size:,} → {total_decompressed_size:,} bytes")
        print(f"📊 Overall expansion: {expansion_ratio:.1f}x")
        print(f"📂 All decompressed files saved to: {output_path.absolute()}")

def list_directory_contents(directory_name):
    """Helper function to see what's in a directory"""
    dir_path = Path(directory_name)
    
    if not dir_path.exists():
        print(f"❌ Directory '{directory_name}' not found!")
        return
    
    print(f"📁 Contents of '{directory_name}':")
    print("-" * 50)
    
    csv_files = []
    gz_files = []
    other_files = []
    
    for item in sorted(dir_path.iterdir()):
        if item.is_file():
            size = item.stat().st_size
            if item.suffix == '.csv':
                csv_files.append((item.name, size))
            elif item.name.endswith('.csv.gz') or item.name.endswith('.gz'):
                gz_files.append((item.name, size))
            else:
                other_files.append((item.name, size))
    
    if gz_files:
        print(f"\n📦 Gzipped files ({len(gz_files)}):")
        for name, size in gz_files:
            print(f"  • {name} ({size:,} bytes)")
    
    if csv_files:
        print(f"\n📄 CSV files ({len(csv_files)}):")
        for name, size in csv_files:
            print(f"  • {name} ({size:,} bytes)")
    
    if other_files:
        print(f"\n📋 Other files ({len(other_files)}):")
        for name, size in other_files:
            print(f"  • {name} ({size:,} bytes)")
    
    if not csv_files and not gz_files and not other_files:
        print("📭 Directory is empty")

def test_csv_file(csv_filename, directory="Unzipped_Results_Per_Section"):
    """Test that a decompressed CSV file can be read properly"""
    csv_path = Path(directory) / csv_filename
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    try:
        print(f"🧪 Testing: {csv_filename}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        print(f"✅ Successfully read {len(rows)} rows")
        if rows:
            print(f"📊 Columns: {len(rows[0])}")
            print(f"📄 First row sample: {rows[0][:5]}...")
            if len(rows) > 1:
                print(f"📄 Second row sample: {rows[1][:5]}...")
        
        # Show file size
        size = csv_path.stat().st_size
        print(f"💾 File size: {size:,} bytes ({size/(1024*1024):.2f} MB)")
        
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Decompress gzipped CSV files')
    parser.add_argument('--source', '-s', default='Results_Per_Section',
                       help='Source directory containing gzipped files (default: Results_Per_Section)')
    parser.add_argument('--output', '-o', default='Unzipped_Results_Per_Section',
                       help='Output directory for decompressed files (default: Unzipped_Results_Per_Section)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Show what would be done without actually doing it')
    parser.add_argument('--list-source', action='store_true',
                       help='List contents of source directory and exit')
    parser.add_argument('--list-output', action='store_true',
                       help='List contents of output directory and exit')
    
    args = parser.parse_args()
    
    if args.list_source:
        list_directory_contents(args.source)
        return
    
    if args.list_output:
        list_directory_contents(args.output)
        return
    
    print("🗜️  GZIP TO CSV DECOMPRESSOR")
    print("=" * 60)
    print(f"📂 Source: {args.source}")
    print(f"📂 Output: {args.output}")
    print(f"🔍 Dry run: {args.dry_run}")
    print("=" * 60)
    
    process_gzipped_files(args.source, args.output, args.dry_run)

if __name__ == "__main__":
    # If run as script, use command line arguments
    main()
else:
    # If imported, provide easy functions for interactive use
    print("📚 gzip_to_csv module loaded!")
    print("Available functions:")
    print("  • process_gzipped_files() - Main decompression function")
    print("  • list_directory_contents() - See what's in a directory")
    print("  • test_csv_file() - Test a specific CSV file")
    print("\nQuick start:")
    print("  1. list_directory_contents('Results_Per_Section')")
    print("  2. process_gzipped_files(dry_run=True)  # Test first")
    print("  3. process_gzipped_files()  # Actually decompress")
