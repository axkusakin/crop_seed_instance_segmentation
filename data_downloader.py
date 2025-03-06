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
    # Path to the current directory
    current_dir = os.getcwd()

    # Path to the ZIP file
    zip_file = os.path.join(current_dir, "data.zip")

    # Path to the output directory (default is "data" or custom name)
    data_dir = os.path.join(current_dir, output_dir)

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

    # Step 2: Download the ZIP file if it doesn't exist
    if not os.path.exists(zip_file):
        print("Downloading data.zip...")
        gdown.download(download_url, zip_file, quiet=False)
        print("Download complete!")

    # Step 3: Extract the ZIP file into the current directory
    if os.path.exists(zip_file):
        print(f"Extracting data.zip to '{data_dir}'...")
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(current_dir)  # Extract to the current directory
            print(f"Extraction complete! Files are stored in '{current_dir}'.")
        except zipfile.BadZipFile:
            print(f"Error: {zip_file} is not a valid ZIP file.")

    # Step 4: Rename the "data" folder (if a custom name is provided)
    default_data_dir = os.path.join(current_dir, "data")  # Path to the default "data" folder
    if os.path.exists(default_data_dir) and output_dir != "data":
        os.rename(default_data_dir, data_dir)  # Rename the folder
        print(f"Renamed 'data' directory to '{output_dir}'.")

    # Step 5: Remove the __MACOSX directory if it exists
    macosx_dir = os.path.join(current_dir, "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)

    # Step 6: Remove files with names like 'Icon'$'\r'
    for root, dirs, files in os.walk(current_dir, topdown=False):
        for file_name in files:
            if "\r" in file_name or file_name.startswith("Icon"):
                file_path = os.path.join(root, file_name)
                os.remove(file_path)  # Remove unwanted files

    # Step 7: Delete the ZIP file after extraction
    if os.path.exists(zip_file):
        os.remove(zip_file)
        print(f"Deleted the ZIP file: {zip_file}")

    print("Process complete!")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Download and extract data.zip from Google Drive.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing directory if it exists."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data",
        help="Custom name for the 'data' folder (default: 'data')."
    )
    args = parser.parse_args()

    # Run the script
    download_and_extract_data(force=args.force, output_dir=args.output_dir)
