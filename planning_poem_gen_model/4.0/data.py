# python
# -*- coding: utf-8 -*-
# file: data.py
# author: Jie Li
# ------------------------------------------------------------------------
import collections
import os
import sys
import numpy as np
import re
import json
import pdb
import copy

class DataHandle(object):
    def __init__(self,poem_file,keyword_file,args):
        self.word2id = dict()   #dict for word2id
        self.id2word = dict()   #dict for id2word
        self._pt = 0            #pointer for batch position
        self.input = []         #list of input(use id), include all previous sentences
        self.keyword = []       #list of keyword(use id), include the keyword of its target sentence.
        self.target= []         #list of target(use id), include the target sentence
        self.args = args        #configures
        self.size = 0           #input/keyword/target size
        self.dictsize = 0       #vocabnulary size
        self.maxkeylen = 0      #max length of keywords(include EOS)

        self.processPoems(poem_file,keyword_file)
        args.keyword_length = self.maxkeylen

        print('[INFO] Vocabulary Size: ', self.dictsize)
        print('[INFO] Input Size:',len(self.input))
        print('[INFO] Keywords Size: ',len(self.keyword))

    def convertWord2Id(self,c):
        """convert a Chinese char to a word id
            input: 
                c - a Chinese char
                    or '$' as EOS
                    or 'UNK' as unknown char
                    or 'NULL' as nothing
            output: 
                id -a int as ID
                    in my trained dictionary:
                    'NULL' is 0
                    'UNK' is 1
                    '$'(EOS) is 2
            For example:

            convertWord2Id('NULL') returns 0
        """
        if c in self.word2id:
            return self.word2id[c]
        else:
            return self.word2id['UNK']

    def convertKey2Id(self,c):
        """convert a keyword to a word id
            input: 
                c - a keyword with length <=self.maxkeylen
            output: 
                id -a int as ID
                    in my trained dictionary:
                    'NULL' is 0
                    'UNK' is 1
                    '$'(EOS) is 2
        """
        if len(c)>=self.maxkeylen:
            print('[DATA ERROR] Max keyword length is',self.maxkeylen-1)
            return None
        listkeyid = [0 for i in range(self.maxkeylen)]
        for ch in range(len(c)):
            listkeyid[ch] = self.convertWord2Id(c[ch])
        listkeyid[len(c)] = 2
        return listkeyid

    def convertId2Word(self,c):
        """convert a word id to a Chinese char
            Reverse of convertWord2Id(...)
            For example:

            convertId2Word(0) returns 'NULL'
        """
        tmp = str(c)
        if tmp in self.id2word:
            return self.id2word[tmp]
        else:
            return self.id2word[str(self.convertWord2Id['UNK'])]

    def processPoems(self,poem_file,keyword_file):
        """initialize the dataset
            Initial dataset to the training format.
        """
        poems = []
        keys = []
        target= []
        lines = []
        with open(poem_file, encoding='utf-8') as f:
            data = f.read()
        with open(keyword_file, encoding='utf-8') as k:
            datak = k.read()
        data=data.replace('\n','$').replace('，',' ').replace('。',' ').replace('：',' ').replace('“',' ').replace('，',' ')
        poems = data.split('$')
        poems = poems[:-1]
        poemslist = []
        keysinpoems = datak.split('\n')
        keysinpoems = keysinpoems[:-1]
        for keysinpoem in keysinpoems:
            for key in keysinpoem.split(','):
                if (len(key)+1)>self.maxkeylen:
                    self.maxkeylen = len(key)+1
                keys.append(key)

        dict_file = 'data/dict.json'
        d = open(dict_file,'r',encoding='utf-8',)
        self.word2id = json.load(d)
        rdict_file = 'data/rdict.json'
        rd = open(rdict_file,'r',encoding='utf-8',)
        self.id2word = json.load(rd)

        self.dictsize = len(self.word2id)

        stopch = self.convertWord2Id('$')
        unkch = self.convertWord2Id('UNK')
        nullch = self.convertWord2Id('NULL')
        sentence_num = self.args.poem_form[0]
        sentence_len = self.args.poem_form[1]+1 #include $ as EOS

        for poem in poems:
            poem += '$'
            poem = poem.replace(' ','$')
            poemId = []
            for ch in poem:
                poemId.append(self.convertWord2Id(ch))
            if len(poemId) == sentence_num*sentence_len:
                poemslist.append(poemId)
            else:
                print('[DATA ERROR] Poem Wrong Format:')
                print(poem)

        #each poem generate 4 [input,target]
        for poem in poemslist:
            input_sentence = [nullch for i in range((sentence_num-1)*sentence_len)]
            for i in range(sentence_num):
                target_sentence = [nullch for i in range(sentence_len)]
                for j in range(sentence_len):
                    if i > 0:
                        input_sentence[(i-1)*sentence_len+j] = poem[(i-1)*sentence_len+j]
                self.input.append(copy.copy(input_sentence))
                for j in range(sentence_len):
                    target_sentence[j] = poem[i*sentence_len+j]
                self.target.append(copy.copy(target_sentence))

        self.size = len(self.input)
        print('[INFO] Poems Size: ',len(poemslist))

        #generate keywords
        for key in keys:
            keylist = [nullch for i in range(self.maxkeylen)]
            for ch in range(len(key)):
                keylist[ch] = self.convertWord2Id(key[ch])
            keylist[len(key)] = stopch
            self.keyword.append(keylist)

    def generateBatch(self):
        """generate the batches
            Once call this function, returns self.args.batch_size samples as a batch.
            Will memorize the begining position of next batch.
            returns [sequence_len batch_size]
        """
        sequence_len = (self.args.poem_form[0]-1)*(self.args.poem_form[1]+1)
        x_batches = [[] for i in range(sequence_len)]
        y_batches = [[] for i in range(self.args.poem_form[1]+1)]
        k_batches = [[] for i in range(self.maxkeylen)]
        for i in range(self.args.batch_size):
            if self._pt + 1 >= self.size:
                self._pt = 0
            for j in range(sequence_len):
                x_batches[j].append(self.input[self._pt][j])
            for j in range(self.args.poem_form[1]+1):
                y_batches[j].append(self.target[self._pt][j])
            for j in range(self.maxkeylen):
                k_batches[j].append(self.keyword[self._pt][j])
            self._pt += 1
        return x_batches, y_batches, k_batches

    def getDictsize(self):
        return self.dictsize

    def debugInput(self, i):
        """returns the ith Input(id)
        """
        if i < self.size:
            return self.input[i]
        else:
            return []