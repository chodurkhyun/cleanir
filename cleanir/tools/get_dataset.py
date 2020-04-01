import requests
import os
from zipfile import ZipFile
import tarfile
from tqdm.autonotebook import tqdm


def download_from_url(url, dst):
    """Download a file from URL

    Arguments:
        url {str} -- url to download file
        dst {str} -- dst place to put the file

    Return:
        File size
    """

    file_size = int(requests.head(url).headers['Content-Length'])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0

    if first_byte >= file_size:
        return file_size

    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(total=file_size, initial=first_byte,
                unit='B', unit_scale=True, desc=url.split('/')[-1])

    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


def download_jaffe(dst_path):
    """Downloads and extracts JAFFE dataset

    Arguments:
        dst_path {str} -- dst place to put the dataset
    """

    url = 'https://zenodo.org/record/3451524/files/jaffedbase.zip'
    zip_path = os.path.join(dst_path, 'jaffe.zip')

    try:
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        download_from_url(url, zip_path)
        with ZipFile(zip_path, 'r') as zip_obj:
            zip_obj.extractall(dst_path)

        os.remove(zip_path)
    except OSError as error:
        print(error)
        print('Something wrong with processing files..')


def download_lfw(dst_path):
    """Downloads and extracts LFW deep funneled dataset and pairs.txt

    Arguments:
        dst_path {str} -- dst place to put the dataset
    """

    data_url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
    label_url = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'

    data_path = os.path.join(dst_path, 'lfw.tgz')
    label_path = os.path.join(dst_path, 'pairs.txt')

    try:
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        download_from_url(label_url, label_path)
        download_from_url(data_url, data_path)
        with tarfile.open(data_path, 'r') as tar_obj:
            tar_obj.extractall(dst_path)

        os.remove(data_path)
    except OSError as error:
        print(error)
        print('Something wrong with processing files..')
