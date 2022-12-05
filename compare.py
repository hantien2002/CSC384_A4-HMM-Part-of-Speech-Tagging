import os
import sys

def read_file(file) -> list:
    file = open(file, "r")
    lst = []
    for x in file:
        x = x.strip()
        lst.append(x)
                
    return lst

def compare(lst1, lst2):
    correctness = 0
    total = len(lst1)
    for i in range(len(lst1)):
        if lst1[i] == lst2[i]:
            correctness += 1
    print(correctness)
    print(total)
    return correctness/total


# python3 compare.py data/training1.txt data/output1.txt

if __name__ == '__main__':
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    
    f1 = read_file(arg1)
    f2 = read_file(arg2)
    
    cor = compare(f1, f2)
    print(cor)
    
    
    