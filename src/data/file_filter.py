import os
import shutil
import argparse

from tqdm import tqdm


# Ignored statoin IDs
IGNORED_STATIONS = [
    'C891'
]

# Station IDs of stations outside the State of São Paulo which data
# will be used (mostly nearby stations)
EXTRA_STATIONS = [
    'A850',
    'A849',
    'S705',
    'A759',
    'A704',
    'S717',
    'A520',
    'A561',
    'A525',
    'A509',
    'A529',
    'A619',
    'A628',
    'A635',
    'A746'
]


def get_station_id(filename: str) -> str:
    """
    Obtains the Station ID from a .csv datafile name.

    Parameters
    -----------
    `filename`: str
        Name of the datafile in filesystem

    Returns
    ---------
    `station_id`: str
        Station ID obtained from the filename
    """

    return filename.split('_')[3]


def select_files(source_dir, verbose):
    """
    Uses the globals IGNORED_STATIONS and EXTRA_STATIONS to filter the
    files in `source_dir` that should be used throughout the transformation
    pipelines.

    Parameters
    ----------
    `source_dir`: str
        Directory origin of the datafiles to be selected

    Returns
    ----------
    `files_to_copy`: list
        List of files to copy from `source_dir`. Each element is a 2-tuple
        in the format (<source_dir>, <file>).
    """

    if verbose:
        print(f'Selecting .csv files from directory \'{source_dir}\'...')

    files_to_copy = []

    # each one of the directories contains the observatino (in all available
    # sites) during that year
    for year in os.listdir(source_dir):
        for file in os.listdir( os.path.join(source_dir, year) ):

            station_id = get_station_id(file)

            if station_id not in IGNORED_STATIONS:
                if 'INMET_SE_SP' in file or station_id in EXTRA_STATIONS:

                    root = os.path.join( os.getcwd(), source_dir, year )
                    files_to_copy.append((root, file))

    return files_to_copy


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description = 'Selects all .csv files relative to State of São Paulo')

    ap.add_argument('-src', '--source_dir',
                    metavar = '',
                    required = True,
                    help = 'Data file source directory.')

    ap.add_argument('-dst', '--destination_dir',
                    metavar = '',
                    required = True,
                    help = 'Selected data file destination directory.')

    ap.add_argument('-v', '--verbose',
                    metavar = '',
                    required = False,
                    default = 0,
                    help = "Verbosity level: 0 (silent) otherwise verbose.",
                    type = int)

    args = vars(ap.parse_args())


    if args['source_dir'] not in os.listdir(os.getcwd()):
        raise KeyError(f'Source directory \'{args["source_dir"]}\' not found in {os.getcwd()}!')

    if args['destination_dir'] not in os.listdir(os.getcwd()):
        os.mkdir(os.path.join(os.getcwd(), args['destination_dir']))

    # performs file selection 
    selected_to_copy = select_files(source_dir = os.path.join(os.getcwd(), args['source_dir']),
                                    verbose = args['verbose'])

    if args['verbose']:
        print(f"Copying selected files to \'{args['destination_dir']}\'...")

    # copies selected files from 'source_dir' to 'destination_dir'
    for source, file in tqdm(selected_to_copy, ascii = True):
        shutil.copyfile(src = os.path.join(source, file),
                        dst = os.path.join(args['destination_dir'], file))

    if args['verbose']:
        print('Files copied successfully!')
