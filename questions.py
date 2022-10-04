import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))
    #rint(query)
    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    #rint(filenames)
    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dic={}
    for file in os.listdir(directory):
        with open(os.path.join(directory,file),encoding='utf8') as f:
            str = f.read().replace('\n', '')
            #print("STR_TYPE",str)
            dic[file]=str
    
    return dic
            
    raise NotImplementedError


def tokenize(s):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
   
    s=s.lower()
    s=word_tokenize(s)
    #nt("s_before: ",s)
    for word in s.copy():
    
        if word in nltk.corpus.stopwords.words("english") or word in string.punctuation:
            s.remove(word)
    
   #print("safter:" ,s)
    return s
    raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    dic={}
    
    for doc in documents:
        s=set(documents[doc])
        
        for word in s:
            if not word in dic.keys():
                dic[word]=1
            else:
                dic[word]=dic[word]+1
            
    l=len(documents)
    for word in dic.keys():
        dic[word]=math.log10(l/dic[word])
    
    return dic
    
    raise NotImplementedError


        
    
def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    In the value files there are repetition
    """
    lst=[]
    dic={}
    
    def put(file):
        for i in range(0,n):
            if i>=len(lst):
                lst.append(file)
                break
            elif dic[lst[i]]<dic[file]:
                lst.insert(i,file)
                break
                
                
    for file in files.keys():
        sum=0
        set1=set(files[file])
        set2=set(query)
        for word in set1.intersection(set2):
            tf=files[file].count(word)
            sum=sum+tf*idfs[word]
        dic[file]=sum
        put(file)
    
    lst=lst[:n]
    return lst
    raise NotImplementedError
 

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    lst=[]
    dic={}
    
    def put(sentence):
        for i in range(0,n):
            if i>=len(lst):
                lst.append(sentence)
                break
            elif dic[lst[i]][0]<dic[sentence][0]:
                lst.insert(i,sentence)
                break
            elif dic[lst[i]][0]==dic[sentence][0]:
                if dic[lst[i]][1]<dic[sentence][1]:
                    lst.insert(i,sentence)
                    break
            
    
        
    for sentence in sentences.keys():
        set1=set(sentences[sentence])
        #print(sentences[sentence])
        set2=set(query)
        set3=set1.intersection(set2)
        sum=0
        for word in set3:
            sum=sum+idfs[word]
        dic[sentence]=(sum,(len(set3)/len(sentences[sentence])))
        put(sentence)
    
    lst=lst[:n]
    return lst
        
    raise NotImplementedError


if __name__ == "__main__":
    main()
