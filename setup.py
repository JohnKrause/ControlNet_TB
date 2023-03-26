import boto3
import tarfile
import os
import json
import threading
from config import *
from itertools import islice
import multiprocessing


def download_from_s3():
    # Load the authentication details from the JSON file
    with open('secrets.secret') as f:
        secrets = json.load(f)
    
    # Connect to the S3 bucket using the authentication details
    s3 = boto3.resource('s3', aws_access_key_id=secrets['aws_access_key'], aws_secret_access_key=secrets['aws_secret_key'])
    bucket = s3.Bucket(secrets['aws_bucket_name'])

    os.makedirs(os.path.dirname(f'training/{REVNUM}'), exist_ok=True)
    os.makedirs(os.path.dirname(f'training/{REVNUM}/models/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'training/{REVNUM}/{CONTROL_TYPE}/images_resize/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'training/{REVNUM}/{CONTROL_TYPE}/controls/'), exist_ok=True)

    #Download the training data...
    download_bucket_file(TRAINDB, TRAINDB_LOCAL, bucket)
    download_bucket_file(MODEL, MODEL_LOCAL, bucket)
    download_bucket_file(IMAGES_TARGET, IMAGES_TARGET_LOCAL, bucket)
    download_bucket_file(CONTROLS_TARGET, CONTROLS_TARGET_LOCAL, bucket)

    #download manifest of images...

    #extract things using multiprocessing
    print(f"Extracting {IMAGES_TARGET}...")
    tar_extract(IMAGES_TARGET_LOCAL,IMAGES_EXTRACT)
    print(f"Extracted {IMAGES_TARGET} successfully!")
    
    print(f"Extracting {CONTROLS_TARGET}...")
    tar_extract(CONTROLS_TARGET_LOCAL,CONTROLS_EXTRACT)
    print(f"Extracted {CONTROLS_TARGET} successfully!")

def download_bucket_file(s3_loc,local_loc,bucket):
    print(f"Downloading {s3_loc} to {local_loc}")
    dl_status =[0, bucket.Object(key=s3_loc).content_length]
    bucket.download_file(s3_loc, local_loc, Callback=lambda chunk: download_progress(chunk, dl_status))
    print(f"Downloaded {s3_loc} successfully!")  

def download_progress(chunk, dl_status):
    dl_status[0]+=chunk
    percent_complete = int((dl_status[0] / dl_status[1]) * 100)
    print(f"Downloaded {dl_status[0]} bytes of {dl_status[1]} bytes ({percent_complete}%)",end='\r')

def tar_extract(archive,local_dir):
    extracted=0
    with tarfile.open(archive,'r:xz') as tar:
        pool = multiprocessing.Pool(initializer=extract_file_initializer, initargs=(archive,local_dir,))
        results = pool.imap_unordered(extract_file, tar, chunksize=100)
        for r in results:
            extracted+=1
            if not extracted%100:
                print(f"Extracted: {extracted}", end='\r')

        pool.close()
        pool.join()

mp_tarfile=None
mp_archive=None
mp_localdir=None
def extract_file_initializer(archive, local_dir):
    global mp_tarfile
    global mp_archive
    global mp_localdir

    mp_archive=archive
    mp_localdir=local_dir
    mp_tarfile=tarfile.open(mp_archive,'r:xz')

def extract_file(member):
    member.name = os.path.basename(member.name)
    mp_tarfile.extract(member,mp_localdir)

if __name__ == "__main__":
    download_from_s3()