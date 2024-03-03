import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

test = json.load(open('/Users/blessingtorsu/Documents/hw2/testing_label.json'))
output = '/Users/blessingtorsu/Documents/hw2/testing_data.txt'
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))