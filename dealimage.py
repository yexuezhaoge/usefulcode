import glob
import os

path = 'traindata/'


def generate_train_and_val(image_path, txt_file):
    with open(txt_file, 'w') as tf:
        for jpg_file in glob.glob(image_path + '*.png'):
            n=len(jpg_file)-3
            print(str(jpg_file[0:n]+'txt'))
            if os.path.exists(str(jpg_file[0:n]+'txt')):
                tf.write(jpg_file + '\n')


generate_train_and_val(path + 'train/', path + 'train.txt')
generate_train_and_val(path + 'val/', path + 'val.txt')