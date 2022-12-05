# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
from typing import List, Dict, Tuple
import numpy as np
import time

np.set_printoptions(threshold=sys.maxsize)

def read_training_file(file) -> list:
    """
    The last sentence in training1.txt
    [('He', 'PNP'), ('got', 'VVD'), ('back', 'AVP'), ('to', 'PRP'), ('his', 'DPS'), ('own', 'DT0'), 
    ('desk', 'NN1'), ('and', 'CJC'), ('sat', 'VVD'), (',', 'PUN'), ('looking', 'VVG'), ('at', 'PRP'),
    ('his', 'DPS'), ('hands', 'NN2'), (',', 'PUN'), ('unable', 'AJ0'), ('to', 'TO0'), ('decide', 'VVI'), 
    ('whether', 'CJS'), ('to', 'TO0'), ('ring', 'VVI'), ('her', 'PNP'), ('up', 'AVP'), (',', 'PUN'), 
    ('or', 'CJC'), ('what', 'DTQ'), ('to', 'TO0'), ('say', 'VVI'), ('.', 'PUN')]
    """
    file = open(file, "r")
    lst = []
    for x in file:
        x = x.strip()
        x = x.split(":")
        
        cur_pair = [] 
        for item in x:
            item = item.strip()
            cur_pair.append(item)
            
        lst.append((cur_pair[0], cur_pair[1]))    
    
    return lst

def read_test_file(file) -> list:
    file = open(file, "r")
    lst = []
    for x in file:
        x = x.strip()
        lst.append(x)
                
    return lst

def write_file(input: List, file):
    with open(file, 'w') as file:
        for line in input:
            file.write(line[0] + " : " + line[1])
            file.write('\n')

def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    
    train_lst = []
    for input_list in training_list:
        train_lst += read_training_file(input_list)

    # print(emission_probability('.', 'PUN', train_lst))
    
    #use set datatype to check how many unique tags are present in training data
    tags = {tag for word, tag in train_lst}
    # print(len(tags))
    # print(tags)
    
    # check total words in vocabulary
    vocab = {word for word, tag in train_lst}
    
    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, next_tag in enumerate(list(tags)):
        for j, cur_tag in enumerate(list(tags)): 
            tags_matrix[i, j] = transition_probability(cur_tag, next_tag, train_lst)[0]/transition_probability(cur_tag, next_tag, train_lst)[1]
    
    # print(tags_matrix)
    
    test_lst = read_test_file(test_file)
    
    viterbi = viterbi_algorithm(test_lst, tags_matrix, train_lst)
    
    write_file(viterbi, output_file)

    #
    # YOUR IMPLEMENTATION GOES HERE
    #



def emission_probability(word: str, tag: str, word_tag_lst: list) -> tuple:
    tag_lst = []
    for pair in word_tag_lst:
        if pair[1] == tag:
            tag_lst.append(pair)
    
    # total number of times the passed tag occurred in word_tag_lst
    count_tag = len(tag_lst) 
    
    word_given_tag_lst = []
    for pair in tag_lst:
        if pair[0] == word:
            word_given_tag_lst.append(pair[0])
    
    # calculate the total number of times the passed word occurred as the passed tag
    count_word_given_tag = len(word_given_tag_lst)
 
    return (count_word_given_tag, count_tag)


def emi_prob_list(word_tag_lst: list) -> dict:
    # tags = {tag for word, tag in word_tag_lst}
    # vocab = {word for word, tag in word_tag_lst}
    dic = {}
    for word, tag in word_tag_lst:
        if (word, tag) not in dic:
            dic[(word, tag)] = emission_probability(word, tag, word_tag_lst)
    return dic

# compute Transition Probability
def transition_probability(cur_tag, next_tag, word_tag_lst):
    tags = []
    for pair in word_tag_lst:
        tags.append(pair[1])
            
    next_tags = []
    for tag in tags:
        if tag == next_tag:
            next_tags.append(tag)
    count_cur_tags = len(next_tags)
    count_next_given_cur = 0
    
    for index in range(len(tags)-1):
        if tags[index]==next_tag and tags[index+1] == cur_tag:
            count_next_given_cur += 1
    return (count_next_given_cur, count_cur_tags)


def viterbi_algorithm(words, trans_mat, word_tag_lst):  
    
    emi_lst = emi_prob_list(word_tag_lst)
      
    states = []
    tags = list(set([pair[1] for pair in word_tag_lst]))
     
    for key, word in enumerate(words):
        route = [] 
        for tag in tags:
            if key == 0:
                #print(trans_mat)
                idx1 = tags.index("PUN")
                idx2 = tags.index(tag)
                trans_prob = trans_mat[idx1][idx2]
            else:
                idx1 = tags.index(states[-1])
                idx2 = tags.index(tag)
                trans_prob = trans_mat[idx1][idx2]
                 
            # emission probabilities
            emi_prob = 0
            if (word, tag) in emi_lst:
                a = emi_lst[word, tag][0]
                b = emi_lst[word, tag][1]
                emi_prob = a/b

            # state probabilities
            state_prob = emi_prob * trans_prob    
            route.append(state_prob)
             
        pmax = max(route)
        # max prob state
        state_max = tags[route.index(pmax)] 
        states.append(state_max)
        # print(state_max)
    return list(zip(words, states))


if __name__ == '__main__':
    t0 = time.time()
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
    t1 = time.time()
    print(time.asctime(time.localtime(t0)))
    print(time.asctime(time.localtime(t1)))
    
    # python3 tagger.py -d validation/training1.txt -t validation/test1.txt -o validation/output1.txt
    # python3 compare.py validation/solution1.txt validation/output1.txt