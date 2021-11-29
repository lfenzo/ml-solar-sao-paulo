import os

from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlretrieve


def download_files(url, dst_dir, period_start, period_end):
    """
    Downloads datafiles from `url` into `dst_dir` in a period starting in
    `period_start` up to `period_end`

    Parameters
    ----------
    `url`: str
        URL from which data will be downloaded.

    `dst_dir`: str
        Destination directory where the downloaded files will be stored.
        Files are temporarily stored in .zip format.

    `period_start`: int
        Staring year of the selected period (first year of observations)

    `period_end`: int
        Last year of the selected period.

    """

    os.chdir(dst_dir)

    for year in tqdm(range(period_start, period_end + 1), ascii = True):

        # checks if donwload of the current file is necessary
        if f'{year}.zip' not in os.listdir():

            dest = os.path.join(os.getcwd(), f'{year}.zip')
            urlretrieve(url = f'{url}/{year}.zip', filename = dest)

    os.chdir('..')


def unzip_downloaded_files(src_dir, dst_dir):
    """
    Unzips the downloaded files in `src_dir` into `dst_dir`.

    Parameters
    ----------
    `src_dir`: str
       Directory containing the downloaded .zip files.

    `dst_dir`: str
       Destination diredctory for the unziped files.

    """

    for _zip in tqdm(os.listdir(src_dir), ascii = True):

        with ZipFile(file = os.path.join(os.getcwd(), f'{src_dir}', _zip), mode = 'r') as zip_file:

            # gets the current file yaer
            year = int(_zip.split('.')[0])

            # zip file of 2020 and 2021 have to be handled differently
            if year >= 2020:
                zip_file.extractall(path = os.path.join(os.getcwd(), dst_dir, str(year)))
            else:
                zip_file.extractall(path = os.path.join(os.getcwd(), dst_dir))


if __name__ == '__main__':

    if not os.path.exists('zips'):
        os.mkdir('zips')

    base_url = 'https://portal.inmet.gov.br/uploads/dadoshistoricos'

    download_files(url = base_url,
                   dst_dir = 'zips',
                   period_start = 2000,
                   period_end = 2021)

    if not os.path.exists('inmet_brasil'):
        os.mkdir('inmet_brasil')

    unzip_downloaded_files(src_dir = 'zips', dst_dir = 'inmet_brasil')

