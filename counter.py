import sys
import glob
import os
import ast
import re
import pandas as pd
from pprint import pprint
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from gensim.models import TfidfModel, LsiModel, LdaModel, doc2vec

df = pd.read_csv("data.csv")

print("class")

print(len(df[df['type'] == "class"]))

print("method")

print(len(df[df['type'] == "method"]))

print("function")

print(len(df[df['type'] == "function"]))

print("total")
print(len(df))