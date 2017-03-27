#python
#-*- coding:utf-8 -*-

"""
    project: CSCI544
    Author:  Jie li
    Date:    2017/3/26
    Version: 0.0.1
    Summary: Generate KeyWords from Input which can be a sequence of string or a file.

"""

from __future__ import unicode_literals
import sys
import codecs
import re
import jieba
import jieba.posseg
import jieba.analyse

from snownlp import SnowNLP

import pdb



class TextRank(object):
    """Summary of class TextRank

        TextRank are used to extract keywords from input.

        Attributes:
            pkg: parameter to control which textrank package to use.
            packageset: a set contain two value: 'snownlp', 'jieba'. it is used to check if the input is valid.
            debug: control if print handling log. default value : False

    """
    def __init__(self,pkg= "snownlp",debug=False):
        """ Init parameter for the TextRank Class.

        Args:
            pkg: parameter to control which textrank package to use.
                 "snownlp" : use package snowNLP (default value)
                 "jieba"   : use package jieba

        Returns:
            None

        """
        self.packageset = set()
        self.debug = debug
        self.packageset.add("snownlp")
        self.packageset.add("jieba")
        if pkg not in self.packageset:
            print "parameter not right, use default setting: snownlp"
            self.pkg = "snownlp"
            return
        self.pkg = pkg

    def changeTextRankPackage(self,pkg):
        """ change TextRank Package for the TextRank Class.

        Args:
            pkg: parameter to control which textrank package to use.
                 "snownlp" : use package snowNLP (default value)
                 "jieba"   : use package jieba

        Returns:
            None

        """
        if pkg not in self.packageset:
            print "parameter not right, didn't change package, use previous package: ",self.pkg
            return
        self.pkg = pkg
    def __textRankHandle(self,text,count):
        """ perform textRank operation

            Generate and return a list of keywords from the input string.


        Args:
            text: a sequence of strings which need to generate keywords.
            count: the number of keywords generated from text.

        Returns:
            a list of keywords generated from text. All the keywords in the list are
            in Descending order according to its weight
            For example:

            ["桃花开","春天"]


        """
        out = []
        if self.pkg =="snownlp":
            snow = SnowNLP(text)
            out = snow.keywords(count)
            #print "snownlp....."
        else:
            for x, w in jieba.analyse.textrank(text,withWeight=True):
                out.append(x)
            #print "jieba....."
        return out[: count if count<len(out) else len(out)]

    def genKeyWordFromText(self,text,count=8):
        """generate keywords list for text
    
            receive a sequence of string and an int as input, generate and a list of keywords for that string

        Args:
            text: a sequence of strings which need to generate keywords.
            count: the number of keywords generated from text. the default number is 8.
        Returns:
            a list of keywords generated from text. All the keywords in the list are
            in Descending order according to its weight
        """
        return self.__textRankHandle(text,count)

    def genKeyWordFromFile(self,inputfile,outputfile,count=1):
        """generate keywords list for string in the inputfile
    
            receive a sequence of string as inputfile name, for each line in that file do the following operation:
            1. split by ":" and discard the fist part which is the topic of the poem.
            2. for the rest part, split by "," or "。" to generate string list.
            3. for each string in the string list, generate the keywords list. The default length of list is 1. if the length > 1, joined all keywords by ":"
            4. write all the keywords to the outputfile in one line, segmented by ","


        Args:
            inputfile: a sequence of strings which represents a file to handle
            outputfile: a sequence of strings which represents a file to write
            count: the number of keywords generated from each string, default value is 1.
        Returns:
            None
        Raises:
            IOError: An error occurred accessing the inputfile.
        """
        linecount = 0
        
        with codecs.open(inputfile,encoding='utf-8') as f:
            poemList = f.readlines()
        outfile = open(outputfile, 'w')
        base = len(poemList)/10
        for num,poem in enumerate(poemList):
            if base!=0 and num%base==0:
                print "finish.......",num/base*10,"%"
            #poem = unicode(poem.strip(),"utf-8")
            poem = poem.strip()
            sentences = re.split(u'：|:',poem)
            if len(sentences)!=2:
                continue
            sublines = re.split(u'，|。',sentences[1])
            if len(sublines[-1]) == 0:
                sublines.pop()
            if len(sublines)%4!=0 and len(sublines)%6!=0:
                continue
            toWrite = []
            for line in sublines:
                out = self.__textRankHandle(line,count)
                if len(out) == 0:
                    if self.debug: print " can not extract keyword from ",line
                    break
                elif len(out) > 1:
                    toWrite.append(u"：".join(out))
                else:
                    toWrite.append(out[0])
            if len(toWrite)!= len(sublines):
                if self.debug: print "can not generate keywords list for poem: ",sentences[0]
            else:
            	out = u"，".join(toWrite)+u'\r'
                outfile.write(out.encode('utf-8'))
                linecount = linecount + 1
        outfile.close()
        print "generate keywords for file ",inputfile," finished"
        print "total ", len(poemList), " poems"
        print "handle ", linecount, "poems"


def test():
    """ Example of how to use the class TextRank
        s = "春天的桃花开了"

        #default setting: use snownlp package
        snowTextRank = TextRank()  

        #use jieba package
        jiebaTextRank = TextRank("jieba")

        # read data from file and write keywords 
        snowTextRank.genKeyWordFromFile("5or7.txt","output.txt") to file.


        #generate keywords list from a sequence of string.

        print snowTextRank.genKeyWordFromText(s) 


    """
    pass

if __name__ == "__main__":
    test()


