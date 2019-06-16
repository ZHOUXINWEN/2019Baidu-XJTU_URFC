import os
import random

def write_to_txt(name, input_list):
    with open(name, 'w') as f:
        for file_path in input_list:
            f.writelines(file_path)
train =[]
val = []

def extract_filename(name):
    with open(name) as f:
        reader = f.readlines()
        random.shuffle(reader)
        return reader


if __name__ == '__main__':
    
    train = extract_filename('train.txt')
    val = extract_filename('val.txt')

    parttrain = train[:24000]
    result = parttrain + val
    print(len(train[24000:]))
    write_to_txt('1_train.txt', result)
    write_to_txt('1_val.txt', train[24000:])
    print(len(result))
