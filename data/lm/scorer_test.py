import os
from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from util.text import Alphabet, UTF8Alphabet
alphabet = UTF8Alphabet()
scorer = Scorer(0.75,1.85,"/content/DeepSpeech-Indo/data/lm/kenlm_tamil.scorer", alphabet)
print(scorer)
