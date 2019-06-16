import os
import random

def write_to_txt(name, input_list):
    with open('train/'+name, 'w') as f:
        for file_path in input_list:
            f.writelines(file_path)
train =[]
val = []

    
with open('train/all.txt') as f:
    reader = f.readlines()
    random.shuffle(reader)
    train = reader[:32000]
    val = reader[32000:]
    write_to_txt('train.txt', train)
    write_to_txt('val.txt', val)
    print(len(val))
