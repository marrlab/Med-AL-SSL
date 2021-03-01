from options.train_options import get_arguments
import os
import datetime

root = '/home/ahmad/thesis/med_active_learning/logs_final'
arguments = get_arguments()
files = os.listdir(root)


def main():
    files_new = [file for file in files if 'class-nums' not in file and 'epoch' not in file and 'ae' not in file]

    to_remove = []

    files_new_w_date = [f'{file.split("-")[1]}-{file.split("-")[2]}' for file in files_new]
    dup = [x for x in files_new_w_date if files_new_w_date.count(x) > 1]
    dup_date = [files_new[i] for i, x in enumerate(files_new_w_date) if files_new_w_date.count(x) > 1]

    dup_date_dic = {x: [] for i, x in enumerate(dup)}

    for i, x in enumerate(dup):
        dup_date_dic[x].append(datetime.datetime.strptime(dup_date[i].split('-')[0], '%d.%m.%Y'))

    for k, v in dup_date_dic.items():
        dup_date_dic[k] = sorted(dup_date_dic[k])

    for k, v in dup_date_dic.items():
        to_remove.extend([f"{date.strftime('%d.%m.%Y')}-{k}" for date in v[:-1]])

    for remove in to_remove:
        os.remove(f"{root}/{remove}")


if __name__ == '__main__':
    main()
