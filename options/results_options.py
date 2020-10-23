import argparse
from os.path import expanduser

home = expanduser("~")
code_dir = 'Med-AL-SSL'

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')

parser.add_argument('--log-path', default=f'{home}/{code_dir}/logs_b_100_n_100/', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
