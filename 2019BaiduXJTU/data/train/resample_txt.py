import os
import random
import math
CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
"""
class_num = [9542, 7538, 3590, 1358, 3464, 5507, 3517, 2617, 2867]
max_num = float(max(class_num))
class_ration = [max_num/i for i in class_num]
"""
def write_to_txt(name, input_list):
    with open(name, 'w') as f:
        for file_path in input_list:
            f.writelines(file_path)
train =[]
val = []


def resample(name, class_num, class_ration):
    final_list = []
    class_to_filename = {i:[] for i in CLASSES}
    with open(name + '.txt') as f:
        reader = f.readlines()
        for i in reader:
            class_to_filename[i[:3]].append(i)
    for i in range(0,9):
        frac = int((class_ration[i] - math.floor(class_ration[i]))*class_num[i])
        integer = int(class_ration[i])
        final_list += class_to_filename[CLASSES[i]]*integer
        if frac != 0 :
           final_list += random.sample(class_to_filename[CLASSES[i]], frac)
        print(frac, integer)
    print(len(final_list))
    write_to_txt('MM_' + name + '.txt', final_list)

def statics(name):

    class_to_filename = {i:[] for i in CLASSES}
    with open(name + '.txt') as f:
        reader = f.readlines()
        for i in reader:
            class_to_filename[i[:3]].append(i)

    num = [len(class_to_filename[i]) for i in CLASSES]
    max_number = float(max(num))
    class_ratio = [max_number/i for i in num]
    return num, class_ratio

if __name__ == '__main__':
    num, class_ratio =statics('1_train')
    print(num, class_ratio)
    resample('1_train', num, class_ratio)
