"""
Data Transformation Pipeline

Composed of the following steps:

0 - (optional) Automated data download (downloads and exctracts
    the files from INMET website). Once the data are downloaded
    this step can be skipped in further executions of this script.

1 - Selects the appropriate files from directory 'inmet_brasil'
    with respect to geograhical localization copying the selected
    files to 'inmet_sao_paulo' directory.

2 - Concatenates the selected files in 'inmet_sao_paulo' into only
    one data file (.csv). During this step header data from the
    selected files are also processed.

3 - Data file preprocessing resulting in site-specific datasets
    stored in the respective directories inside 'stations/'


Depends on/runs the following scripts:

- file_concatenator.py
- file_downloader.py
- file_filter.py
- preprocessing.py


Data Imputation

For the imputation process, the last step (preprocessing) is performed
twice: once with the imputation and once without the imputation. The
produced datasets and objects are stored appropriately inside 'mixer/'
in the appropriate directories.

"""


import os
import argparse


def format_command(script: str, arg_list: list) -> str:
    """
    Formats the arguments passed as a dictionary in such way that they
    are ready to be passed to `os.system(...)`
    """

    command = f'python {script}'

    for arg in arg_list:
        command += f' {str(arg)}'

    return command


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description = "Data Transformation Pipeline: from " + \
                                               "automated download through normalization.")

    ap.add_argument('-dwn', '--download-files',
                    metavar = '',
                    required = False,
                    default = 0,
                    help = 'Automaticaly downloads data files from INMET. Default: 0')

    ap.add_argument('-idw', '--idw-imputation',
                    metavar = '',
                    required = False,
                    default = 0,
                    help = 'Applies IDW data imputation during preprocessing.' \
                           'When (1) is passed both original and imputed datasets ' \
                           'produced.')

    ap.add_argument('-v', '--verbose',
                    metavar = '',
                    required = False,
                    default = 0,
                    help = "Verbosity level: 0 (silent) otherwise verbose.",
                    type = int)

    args = vars(ap.parse_args())


    pipeline_operations = [

        # performs site filtering (selects only the ones in SP State)
        {
            'script': 'file_filter.py',
            'args': [
                '-src', 'inmet_brasil',
                '-dst', 'inmet_sao_paulo',
                '-v', args['verbose'],
            ]
        },

        # concatenates all selected files into only one dataframe
        {
            'script': 'file_concatenator.py',
            'args': [
                '-src', 'inmet_sao_paulo',
                '-v', args['verbose'],
            ]
        },

        # applies preprocessing steps saving all site-speciifc data
        #in the respective directories
        {
            'script': 'preprocessing.py',
            'args': [
                '-i', 'concatenated_dataframe.ftr',
                '-nrm', 'rbst',
                '-idw', 0, #no imputation -> original data
                '-v', args['verbose'],
            ]
        },
    ]

    # adding preprocessing WITH IMPUTATION to the operation pipeline
    if args['idw_imputation']:
        pipeline_operations.append(
            {
                'script': 'preprocessing.py',
                'args': [
                    '-i', 'concatenated_dataframe.ftr',
                    '-nrm', 'rbst',
                    '-idw', 1, # imputed data
                    '-v', args['verbose'],
                ]
            }
        )

    # Adds the optional data download step into the pipeline as the first
    # stop. (should be skipped once all files are already downloaded)
    if args['download_files']:

        pipeline_operations.insert(0, {
            'script': 'file_downloader.py',
            'args': [], # no args for this script
        })

    # executes all the quered pipeline operatoin steps
    for step in pipeline_operations:
        os.system(command = format_command(script = step['script'], arg_list = step['args']))
