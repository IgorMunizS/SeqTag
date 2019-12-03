import re
import config

def clean_numbers(x):

    tokens = x.split()
    for i,token in enumerate(tokens):
        if token.isdigit():
            tokens[i] = config.number_token
    return ' '.join(tokens)

def clean_text(x):

    x = str(x)
    for punct in "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&':
        x = x.replace(punct, config.punct_token)
    return x
