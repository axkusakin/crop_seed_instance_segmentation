#!/usr/bin/env python3
"""
Data Downloader

This script downloads a .zip file from Google Drive, extracts its contents
into a specified directory, and performs cleanup of unnecessary files.
The output directory can be customized, and existing directories can be
overwritten using the --force flag.
"""

import os
import zipfile
import shutil
import argparse
import gdown

def download_and_extract_data(force=False, output_dir="data"):
    # Define paths
    data_dir = os.path.join(os.getcwd(), output_dir)  # Path to the custom output directory
    zip_file = "data.zip"

    # Google Drive file URL
    download_url = "https://drive.google.com/uc?export=download&id=1g8bg9ter9DlKWgs0lfPZMQemRlzRVOQr"

    # Step 1: Check if the output directory already exists
    if os.path.exists(data_dir):
        if not force:
            print(f"Directory '{data_dir}' already exists. Use --force to overwrite.")
            return
        else:
            print(f"Directory '{data_dir}' already exists. Overwriting...")
            shutil.rmtree(data_dir)  # Remove the existing directory

    # Step 2: Download the ZIP file using gdown if it does not exist
    if not os.path.exists(zip_file):
        print("Downloading data.zip...")
        gdown.download(download_url, zip_file, quiet=False)
        print("Download complete!")

    # Step 3: Extract the ZIP file directly into the renamed directory
    if os.path.exists(zip_file):
        print(f"Extracting data.zip to '{data_dir}'...")
        try:
            os.makedirs(data_dir, exist_ok=True)  # Create the output directory if it doesn't exist
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(data_dir)  # Extract directly into the custom directory
            print(f"Extraction complete! Files are stored in '{data_dir}'.")
        except zipfile.BadZipFile:
            print(f"Error: {zip_file} is not a valid ZIP file.")

    # Step 4: Remove the __MACOSX directory if it exists
    macosx_dir = os.path.join(data_dir, "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)

    # Step 5: Remove files with names like 'Icon'$'\r'
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for file_name in files:
            if "\r" in file_name or file_name.startswith("Icon"):
                file_path = os.path.join(root, file_name)
                os.remove(file_path)  # Remove unwanted files

    # Step 6: Delete the ZIP file after extraction
    if os.path.exists(zip_file):
        os.remove(zip_file)
        print(f"Deleted the ZIP file: {zip_file}")

    print("Process complete!")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Download and extract data.zip from Google Drive.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it exists."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data",
        help="Custom name for the output directory (default: 'data')."
    )
    args = parser.parse_args()

    # Run the script
    download_and_extract_data(force=args.force, output_dir=args.output_dir)
