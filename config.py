### BEST CONFIG SO FAR ####
#Fold 1: 93.37
#Fold Pred: 93.651

#data
data_folder = '../dados/'
original_data_folder = '../../sequence_tagging/data/'


#tokens
pad_seq_tag = 'pad'

unk_token = 'desc'
punct_token = 'pontuação'
number_token = 'número'

#emgeddings
glove_file = '/opt/harold/word_embeddings/portugues/glove_s600.txt'
glove_size = 600

fasttext_file = '/opt/harold/word_embeddings/portugues/fast_skip_s600.txt'
fasttext_size = 600

wang_file = '/opt/harold/word_embeddings/portugues/wang_skip_s600.txt'
wang_size = 600

char_size = 100

# training
train_embeddings    = False
nepochs             = 20
dropout             = 0.3
batch_size          = 64
lr_method           = "adam"
lr                  = 0.001
lr_decay            = 0.9
clip                = -1  # if negative, no clipping
nepoch_no_imprv     = 3

#kaggle files
sample_submission = '../../sequence_tagging/data/sample_submission.csv'

#folds
nfolds= 5
seed = 133

# general config
dir_output  = "results/test/"
dir_model   = dir_output + "model.weights/"
path_log    = dir_output + "log.txt"

# embeddings
dim_word = 50
dim_char = 100

# glove files
filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
# trimmed embeddings (created from glove_filename with build_data.py)
filename_trimmed = "data/glove_s{}.trimmed.npz".format(dim_word)
use_pretrained = True

# dataset
filename_train   = "data/dataset/train.conll"
filename_test    = "data/dataset/test.conll"
filename_valid   = "data/dataset/valid.conll"

# filename_dev = filename_test = filename_train = "data/test.txt" # test

max_iter = None  # if not None, max number of examples in Dataset

# vocab (created from dataset with build_data.py)
filename_words = "data/words.txt"
filename_tags  = "data/tags.txt"
filename_chars = "data/chars.txt"



# model hyperparameters
hidden_size_char    = 128  # lstm on chars
hidden_size_lstm    = 256  # lstm on word embeddings

# NOTE: if both chars and crf, only 1.6x slower on GPU
use_crf     = True  # if crf, training is 1.7x slower on CPU
use_chars   = True  # if char embedding, training is 3.5x slower on CPU
