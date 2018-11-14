#!/bin/bash

# copy scripts over
#cp ../WikiQA/gen_w2v.py ../WikiQA/norm_embed.py ../WikiQA/gen_hist4drmm.py ./

# transfer data to mz format
python prepare_mz_data.py

# embedd 
python gen_w2v.py glove.6B.300d.txt word_dict.txt embed_glove_d300
python norm_embed.py embed_glove_d300 embed_glove_d300_norm

cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf
python gen_hist4drmm.py 60


