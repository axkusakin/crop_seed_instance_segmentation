#!/usr/bin/env python3
"""
Data Downloader

This script downloads a .zip file from Google Drive, extracts its contents
into the 'data' directory, and performs cleanup of unnecessary files.
Existing directories can be overwritten using the --force flag.
"""

import os
import argparse
import gdown
import zipfile
import shutil

# Google Drive file URL
download_url = "https://drive.google.com/uc?export=download&id=1g8bg9ter9DlKWgs0lfPZMQemRlzRVOQr"

def download_and_extract(force):
    """Download data.zip, extract its contents to the current directory, and clean up unnecessary files."""
    zip_file = "data.zip"
    folder_name = "data"
    
    # Check if data.zip or extracted contents already exist
    if os.path.exists(zip_file) or os.path.exists(folder_name):
        if not force:
            print(f"Error: {zip_file} or {folder_name} already exists. Use --force to overwrite.")
            return
        else:
            if os.path.exists(zip_file):
                os.remove(zip_file)
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
            print("Overwriting existing files...")
    
    # Download the zip file
    print(f"Downloading {zip_file}...")
    gdown.download(download_url, zip_file, quiet=False)
    print("Download complete!")

    # Extract the zip file
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(".")  # Extract directly into the current directory
    print("Extraction complete!")

    # Delete the zip file
    os.remove(zip_file)
    print(f"Deleted the ZIP file: {zip_file}")
    
    # Remove __MACOSX directory if it exists
    macosx_dir = os.path.join(".", "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)

    # Remove files like 'Icon'$'\r'
    for root, _, files in os.walk(".", topdown=False):
        for file_name in files:
            if "\r" in file_name or file_name.startswith("Icon"):
                file_path = os.path.join(root, file_name)
                os.remove(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract barley data from Google Drive.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if they exist")
    args = parser.parse_args()

    download_and_extract(args.force)
