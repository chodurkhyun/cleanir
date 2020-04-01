import gdown
import os
from zipfile import ZipFile


def get_cleanir_model(dst_path):
    """Downloads and extracts CLEANIR's models

    Arguments:
        dst_path {str} -- dst place to put the models
    """

    url = 'https://drive.google.com/uc?id=1sEVj4hpM2xWEc3E3XWKMCOyv4HQv3zJI'
    file_path = os.path.join(dst_path, 'cleanir_model.zip')

    try:
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        gdown.download(url, file_path, quiet=False)
        with ZipFile(file_path, 'r') as zip_obj:
            zip_obj.extractall(dst_path)

        os.remove(file_path)
    except OSError as error:
        print(error)
        print('Something wrong with processing files..')
