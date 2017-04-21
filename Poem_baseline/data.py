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
import pdb

class DataHandle(object):
    def __init__(self,file_name,args):
        self.word2id = dict()
        self.id2word = dict()
        self._pt = 0
        self.poems = []
        self.output= []
        self.vocab = []
        self.vocab_size = 0
        self.args = args
        self.size = 0
        self.data = []
        self.words= []


        self.process_poems(file_name)
        

    def save_Dict(self,file):
        with open(file,'w') as f:
            f.write('id\tword\n')
            for i in self.id2word:
                x = str(i)+"\t"+self.id2word[i]+"\n"
                f.write(x)

    def convertWord2Id(self,c):
        return self.word2id[c]

    def convertId2Word(self,c):
        return self.id2word[c]

    def process_poems(self,file_name):
        # 诗集
        poems = []
        output= []
        lines = []
        count = 0
        pcount = 0
        with open(file_name, encoding='utf-8') as f:
            self.data = f.read()
        self.size = len(self.data)
        """
        with open(file_name, "r", encoding='utf-8', ) as f:
            for line in f.readlines():
                pcount+=1
                try:
                    title, content = line.strip().split(':')
                    content = content.replace(' ', '')
                    lines.append(title+"："+content)
                    # if len(content) < 5 or len(content) > 79:
                    #     continue
                    contents = re.split('，|。',content)
                    if len(contents[-1])==0: contents.pop()
                    poems.append(title+"：")
                    output.append(contents[0]+"：")
                    for i,x in enumerate(contents[1:]):
                        output.append(x+"。")
                        poems.append(contents[i]+"。")
                except ValueError as e:
                    pass


        # 统计每个字出现次数
        wordset = set()

        all_words = []
        #for poem in lines:
        #    all_words += [word for word in poem]
        # 这里根据包含了每个字对应的频率
        #counter = collections.Counter(all_words)
        #count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        #words, _ = zip(*count_pairs)
        for line in poems:
            for x in line:
                wordset.add(x)
        for line in output:
            for  x in line:
                wordset.add(x)
        wordset.add(' ')
        words = list(wordset)
        words.sort()



        # 取前多少个常用字
        #words = words[:len(words)] + (' ',)
        """
        self.total_len = len(self.data)  # total data length
        wordset = set(self.data)
        wordset.add(' ')
        self.words = list(wordset)
        self.words.sort()
        # vocabulary
        self.vocab_size = len(self.words)  # vocabulary size
        print('Vocabulary Size: ', self.vocab_size)

        #self.poems = poems
        #self.output = output

        self.vocab = self.words
        self.word2id = {w:i for i,w in enumerate(self.vocab)}
        self.id2word = {i:w for i,w in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        print('[INFO] Reading Poem: ', pcount)
        print('[INFO] Vocabulary Size: ', self.vocab_size)
        print('[INFO] Input Data Size: ',len(self.poems))
        print('[INFO] Output Data Size: ',len(self.output))
        #self.size = len(self.poems)
        self.save_Dict(self.args.metadata)


   


    def generate_batch(self):
        #n_chunk = len(poems_vec) // batch_size
        x_batches = []
        y_batches = []
        '''
        for i in range(self.args.batch_size):
            if self._pt == self.poems:
                self._pt = 0
            #x_data = np.full(self.args.seq_length,self.word2id[' '],np.int32)
            #y_data = np.full(self.args.seq_length,self.word2id[' '],np.int32)
            x_data = [ self.word2id[' '] for i in range(self.args.seq_length)]
            y_data = [ self.word2id[' '] for i in range(self.args.seq_length)]
            xd = self.poems[self._pt]
            yd = self.poems[self._pt]
            #if(len(xd)>20):
            #    pdb.set_trace()
            for j in range(len(yd)):
                x_data[j] =self.word2id[xd[j]]
            #x_data = self.poems[self._pt]
                y_data[j] =self.word2id[yd[j]]
            #y_data = self.output[self._pt]
            self._pt+=1
            x_batches.append(x_data)
            y_batches.append(y_data)
            #print(x_data)
        return x_batches, y_batches
        '''
        for i in range(self.args.batch_size):
            if self._pt + self.args.seq_length + 1 >= self.size:
                self._pt = 0
            bx = self.data[self._pt: self._pt+ self.args.seq_length]
            by = self.data[self._pt +
                           1: self._pt + self.args.seq_length + 1]
            self._pt += self.args.seq_length  # update pointer position

            # convert to ids
            bx = [self.word2id[c] for c in bx]
            by = [self.word2id[c] for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches