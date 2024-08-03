"""
A script for downloading a file from S3 bucket
"""

import boto3
import os

S3_BUCKET_NAME = 'y-data-fido-project'
LOCAL_DIR = '../downloads'


def downloading_image_from_s3(img_uri: str,
                              key_id: str, secret_access_key: str,
                              local_dir: str,
                              bucket_name: str = S3_BUCKET_NAME):

    """
    Downloading
    :param img_uri: URI of the image
    :param key_id: credentials for accessing S3
    :param secret_access_key: credentials for accessing S3
    :param local_dir: path to the directory where you want to store the file
    :param bucket_name: bucket where the image is stored

    :returns: saves image locally, returns path to saved image
    """

    # creating the directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    # initializing S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_access_key
    )
    print(f'Connected to S3')

    # image name
    img_name = img_uri.replace('/', '_')
    save_to_path = os.path.join(local_dir, img_name)

    try:
        s3_client.download_file(bucket_name, img_uri, save_to_path)
        print(f'Image downloaded successfully: {img_name}')
        return save_to_path, img_name
    except Exception as e:
        print(f'Error: {str(e)}')
        return None, None
