import os, sys, optparse

import torch
import tqdm
import pymagnitude
import heapq
from scipy.spatial.distance import cosine
import numpy as np

CURDIR = os.path.dirname(os.path.realpath(__file__))

class LexSub:
    def __init__(self, wvec_file, topn=10, train=True):
        self.topn = topn
        self.wv = {}
        
        retrofit_path = os.path.join(CURDIR, '..', 'data', 'glove.6B.100d.retrofit.magnitude')
        if(train or not os.path.isfile(retrofit_path)):
            self.train_vec(wvec_file)
        self.wvecs = pymagnitude.Magnitude(os.path.join(CURDIR, '..', 'data', 'glove.6B.100d.retrofit.magnitude'))

    def train_vec(self, wvec_file):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.times = 2
        self.wv = {}
        self.lexiconDict = {}
        lexiconFile = open(os.path.join(CURDIR, '../data/lexicons/ppdb-xl.txt'), 'r').readlines()
        convertLexiconFileToDict = tqdm.tqdm(lexiconFile, total=len(lexiconFile))
        convertLexiconFileToDict.set_description("Converting lexicon file to dict")
        for line in convertLexiconFileToDict:
            line = line.split()
            temp = []
            for word in line:
                if word not in self.wvecs:
                    continue
                temp.append(word)
            for word in line:
                if word not in self.wvecs:
                    continue
                if word not in self.lexiconDict:
                    self.lexiconDict[word] = {}
                for word2 in temp:
                    if word2 == word:
                        continue
                    if word2 in self.lexiconDict[word]:
                        self.lexiconDict[word][word2] += 1
                    else:
                        self.lexiconDict[word][word2] = 1

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        convert = tqdm.tqdm(self.wvecs, total=len(self.wvecs))
        convert.set_description("Convert numpy array to torch tensor")
        for key, vector in convert:
            self.wv[key] = torch.from_numpy(vector)
            self.wv[key] = self.wv[key].to(device)

        iter = tqdm.tqdm(range(self.times), total=len(range(self.times)))
        iter.set_description("Iteration")
        for i in iter:
            word_progress = tqdm.tqdm(self.wv.keys(), total=len(self.wv.keys()))
            word_progress.set_description("Progress of each iteration")
            for key in word_progress:
                processedVector = self.adjustVector(key, self.wv[key], device)
                self.wv[key] = processedVector

        with open(os.path.join(CURDIR, 'txtFile.txt'), 'w') as f:
            porting = tqdm.tqdm(self.wv.keys(), total=len(self.wv.keys()))
            porting.set_description("Port key and vectors to txt file")
            for key in porting:
                list = self.wv[key].tolist()
                if len(list) != 100:
                    porting.set_description("Length is not 100")
                print(key, file=f, end=" ")
                for i in list:
                    print(i, file=f, end=" ")
                print("", file=f)

        save_path = os.path.join(CURDIR, '../data/glove.6B.100d.retrofit.magnitude')
        os.system('python -m pymagnitude.converter -i txtFile.txt -o '
                  + save_path)

    def adjustVector(self, key, vector, device):
        if key not in self.lexiconDict:
            return vector
        relatedWords = self.lexiconDict[key]
        if len(relatedWords) == 0:
            return vector
        alpha = 0.5
        beta = 0.5
        result = torch.zeros(len(vector))
        result = result.to(device)

        # for word in relatedWords:
        #     wordVec = self.wv[word]
        #     result = torch.add(result, torch.add(torch.mul(vector, beta), torch.mul(wordVec, alpha)))
        # result = torch.mul(result, 1 / ((len(relatedWords)) * (alpha + beta)))
        # return result

        count = 0
        for word in relatedWords:
            wordVec = self.wv[word]
            for i in range(relatedWords[word]):
                count += 1
                result = torch.add(result, torch.add(torch.mul(vector, beta), torch.mul(wordVec, alpha)))
        result = torch.mul(result, 1 / ((count) * (alpha + beta)))
        return result

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        
        # k = 2
        # ks = max(0, index - k)
        # ke = min(len(sentence) - 1, index + k)
        # cs = 1
        # tcs = []
        # for i in range(ks, ke):
        #     if i == index:
        #         continue
        #     tcs.append(sentence[i])
        #     cs += 1
        
        # cvecs = []
        # tvec = self.wvecs.query(sentence[index])
        # for c in tcs:
        #     cvecs.append(self.wvecs.query(c))
        # # cvecs = np.delete(cvecs, index, axis=0)
        # cvecs = np.array(cvecs)
        # combined = ( 0.9 * tvec + 0.1 * np.sum(cvecs, axis=0)) / cs
        # topn = [ k[0] for k in self.wvecs.most_similar(combined, topn=self.topn + 1)]
        # if(sentence[index] in topn):
        #     topn.remove(sentence[index])
        # return (list(topn)[0:10])
        return (list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))


if __name__ == '__main__':
    
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join(CURDIR, '..', 'data', 'input', 'dev.txt'),
                         help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile",
                         default=os.path.join(CURDIR, '..', 'data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-t", "--train", dest="train", default=False, help="Indicate whether to train the word vector or not")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexsub = LexSub(opts.wordvecfile, int(opts.topn), bool(opts.train))
    num_lines = sum(1 for line in open(opts.input, 'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
