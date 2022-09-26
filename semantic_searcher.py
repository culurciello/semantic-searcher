# text similarity
# inspired by: https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python

import sys
import numpy as np
from sentence_transformers import SentenceTransformer, util

print('Loading SentenceTransformer...')
model = SentenceTransformer('stsb-roberta-large')
print('Done!')

np.set_printoptions(precision=2)

def match_text(sentences_all, input_s):
    print('Embedding all possible sentences...')
    embeddings = model.encode(sentences_all, convert_to_tensor=True)
    print('Embedding input sentence')
    embedinput = model.encode(input_s, convert_to_tensor=True)
    print('Done!')

    similarity = []
    for i, line in enumerate(sentences_all):
        # print(line, input_command)
        s = util.pytorch_cos_sim(embeddings[i], embedinput).item()
        similarity.append(s)
        # print(line, s)

    similarity_sorted = np.sort(similarity)[::-1]
    sorted_idx = np.argsort(similarity)[::-1]
    # print(similarity_sorted, sorted_idx) 
    
    return similarity_sorted, sorted_idx


if __name__ == '__main__':

    if len(sys.argv)>1:
        input_s = sys.argv[1]
    else:
        input_s = "cosine similarity"

    if len(sys.argv)>2:
        sentences_all_file = sys.argv[2]
    else:
        sentences_all_file = "example.txt"



    # read all lines in text file and add them to "sentences_all"
    sentences_all = []
    with open(sentences_all_file) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if line != "\n":
                # print("Line {}: {}".format(cnt, line.strip()))
                sentences_all.append(line)
                cnt += 1

            line = fp.readline()


    print('You are comparing: "'+ input_s + '" to all sentences in: '+sentences_all_file)

    similarity, sorted_idx = match_text(sentences_all, input_s)

    print("Comparing results:")
    for i,value in enumerate(similarity):
        print('{:.2f}'.format(similarity[i]), sentences_all[sorted_idx[i]])
