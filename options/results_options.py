import argparse
from os.path import expanduser

home = expanduser("~")

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')

parser.add_argument('--log-path', default=home+'/med_active_learning/logs_b_100_n_100', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
