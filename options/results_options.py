import argparse
from os.path import expanduser

home = expanduser("~")
code_dir = 'med_active_learning'

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')

parser.add_argument('--log-path', default=f'{home}/{code_dir}/logs_matek_final/', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
