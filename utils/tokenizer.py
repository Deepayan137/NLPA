from operator import add
import re

def tokenize(iota):
    punctuations = ' ,."\'/%();:!-?\n'
    tokens = re.split('[%s]'%(punctuations), iota)
    return tokens
