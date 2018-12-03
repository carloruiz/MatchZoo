import json
import sys
import hashlib
sys.path.append('../../matchzoo/inputs/')
sys.path.append('../../matchzoo/utils/')

from preparation import Preparation
from preprocess import Preprocess, NgramUtil

data_dir = './HotpotQACorpus/'
srcdir = './'
dstdir = './'


def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict

def filter_triletter(tri_stats, min_filter_num=5, max_filter_num=10000):
    tri_dict = {}
    tri_stats = sorted(tri_stats.items(), key=lambda d:d[1], reverse=True)
    for triinfo in tri_stats:
        if min_filter_num <= triinfo[1] <= max_filter_num:
            if triinfo[0] not in tri_dict:
                tri_dict[triinfo[0]] = len(tri_dict)
    return tri_dict


q_count = 0
d_count = 0

def unpack_data(data, corpus_file, rel_file, dids, qty=500):
    i = 0
    q_counter = q_count
    d_counter = d_count

    for entry in data:
        question = entry["question"][:-1] # remove '?'
        corpus_file.write(q_str % (entry["_id"], question))
        for wiki_doc in entry["context"]:
            # format corpus in mz format
            title, text = wiki_doc[0], wiki_doc[1]
            doc = ''.join(text).replace("\"", "").replace("'s"," 's")
            did = hashlib.sha1(title.encode('utf-8')).hexdigest()
            if did not in dids:
                dids.add(did)
                corpus_file.write(d_str % (did, doc))
            # format relations in mz format
            if any([wiki_doc[0] == fact[0] for fact in entry["supporting_facts"]]):
                rel_file.write(rel_str % (1, entry["_id"], did))
            else:
                rel_file.write(rel_str % (0, entry["_id"], did))
            d_counter+=1

        q_counter+=1
        i+=1
        if i == qty:
            break
  
    return q_counter, d_counter

in_files = ['hotpot_train_v1.1.json', 'hotpot_dev_distractor_v1.json', 'hotpot_dev_fullwiki_v1.json']
in_files = [data_dir + f for f in in_files]

out_files = ['relation_train.txt', 'relation_valid.txt', 'relation_test.txt']
out_files = [dstdir + f for f in out_files] 

with open(dstdir+'corpus.txt', 'w') as corpus_file:
    q_str = 'Q%s %s\n'
    d_str = 'D%s %s\n'
    rel_str = '%d Q%s D%s\n'   
  
    dids = set()
    quantities = [2000, 800, 800]
    i = 0
    for in_file_path, out_file_path in zip(in_files, out_files):
        with open(in_file_path) as in_file, open(out_file_path, 'w') as rel_file: 
            print("Loading data from %s" % in_file_path)
            data = json.loads(in_file.read())
            print("SO MUCH DATA", len(data))
            q_counter, d_counter = unpack_data(data, corpus_file, rel_file, dids, qty=quantities[i])

            q_count=q_counter
            d_count=d_counter
        i+=1    
    
preprocessor = Preprocess(word_stem_config={'enable': False}, word_filter_config={'min_freq': 2})
dids, docs = preprocessor.run(dstdir + 'corpus.txt')
preprocessor.save_word_dict(dstdir + 'word_dict.txt', True)
preprocessor.save_words_stats(dstdir + 'word_stats.txt', True)

fout = open(dstdir + 'corpus_preprocessed.txt', 'w')
for inum, did in enumerate(dids):
    fout.write('%s %s %s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum]))))
fout.close()
print('Preprocess finished ...')

# dssm_corp_input = dstdir + 'corpus_preprocessed.txt'
# dssm_corp_output = dstdir + 'corpus_preprocessed_dssm.txt'
word_dict_input = dstdir + 'word_dict.txt'
triletter_dict_output = dstdir + 'triletter_dict.txt'
word_triletter_output = dstdir + 'word_triletter_map.txt'
word_dict = read_dict(word_dict_input)
word_triletter_map = {}
triletter_stats = {}
for wid, word in word_dict.items():
    nword = '#' + word + '#'
    ngrams = NgramUtil.ngrams(list(nword), 3, '')
    word_triletter_map[wid] = []
    for tric in ngrams:
        if tric not in triletter_stats:
            triletter_stats[tric] = 0
        triletter_stats[tric] += 1
        word_triletter_map[wid].append(tric)
triletter_dict = filter_triletter(triletter_stats, 5, 10000)
with open(triletter_dict_output, 'w') as f:
    for tri_id, tric in triletter_dict.items():
        print(tri_id, tric, file=f)
with open(word_triletter_output, 'w') as f:
    for wid, trics in word_triletter_map.items():
        print(wid, ' '.join([str(triletter_dict[k]) for k in trics if k in triletter_dict]), file=f)

print('Triletter Processing finished ...')




