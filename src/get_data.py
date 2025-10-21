import os
import gzip
import shutil
import requests

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/unzipped", exist_ok=True)

files = [
    "StormEvents_details-ftp_v1.0_d2014_c20250520.csv.gz",
    "StormEvents_details-ftp_v1.0_d2015_c20250818.csv.gz",
    "StormEvents_details-ftp_v1.0_d2016_c20250818.csv.gz",
    "StormEvents_details-ftp_v1.0_d2017_c20250520.csv.gz",
    "StormEvents_details-ftp_v1.0_d2018_c20250520.csv.gz",
    "StormEvents_details-ftp_v1.0_d2019_c20250520.csv.gz",
    "StormEvents_details-ftp_v1.0_d2020_c20250702.csv.gz",
    "StormEvents_details-ftp_v1.0_d2021_c20250520.csv.gz",
    "StormEvents_details-ftp_v1.0_d2022_c20250721.csv.gz",
    "StormEvents_details-ftp_v1.0_d2023_c20250731.csv.gz",
    "StormEvents_details-ftp_v1.0_d2024_c20250818.csv.gz",
    "StormEvents_details-ftp_v1.0_d2025_c20250818.csv.gz",
]

base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

for filename in files:
    url = base_url + filename
    dest = os.path.join("data/raw", filename)
    unzipped_dest = os.path.join("data/unzipped", filename.replace(".gz", ""))
    
    print(f"Downloading {filename} ...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {dest}")
        
        # Now unzip it
        print(f"Unzipping {filename} ...")
        with gzip.open(dest, "rb") as f_in:
            with open(unzipped_dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted to {unzipped_dest}")
    else:
        print(f"Failed to download {filename} (status {r.status_code})")
