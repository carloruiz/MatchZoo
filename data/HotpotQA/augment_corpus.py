from collections import defaultdict
import random
from heapq import heappush, heappushpop
from tqdm import tqdm


# only stores positive relations
def load_relations(filename):
    query_relation = defaultdict(list)
    with open(filename) as relations:
        for line in relations:
            flag, query_id, doc_id = line.split()
            if int(flag):
                query_relation[query_id].append(doc_id)
            
    print("Finished reading relations.")
    return query_relation


def load_data(filename):
    docs = []
    queries = {}
    with open('corpus_preprocessed.txt') as corpus:
        for line in corpus:
            toks = line.split()
            _id  = toks[0]
            text = toks[2:]

            if _id[0] != 'D':
                curr_query = _id
                queries[_id] = text
                continue
            
            docs.append((_id, text))
    print("Loaded queries and docs corpus_preprocessed.txt")
    return queries, docs


def load_idfs(filename):
    idfs = {}
    with open(filename) as embed_idfs:
        for line in embed_idfs:
            word, idf = line.split()
            idfs[word] = idf
    print("Loaded inverse document frequencies from embed.idf")
    return idfs


def sort(x):
    x.sort(); return x





input_files = ["relation_train.txt", "relation_valid.txt", "relation_test.txt"]
output_files = ["relation_train_augmented.txt", "relation_valid_augmented.txt", "relation_test_augmented.txt"]

doc_num = 100


if __name__ == "__main__":
    corpus_file = "corpus_preprocessed.txt"
    idf_file = "embed.idf"


    queries, docs = load_data(corpus_file)
    idfs = load_idfs(idf_file)

    for relation_file, out_file in zip(input_files, output_files):
        f = open(out_file, 'w') 
        relations = load_relations(relation_file)
        for query_id in tqdm(sort(list(relations.keys()))):
            heap = []
            query_words = set(queries[query_id])
            for doc in docs:
                word_freqs = defaultdict(int)
                for word in doc[1]:
                    if word in query_words: word_freqs[word]+=1
                tfidf = sum([freq*float(idfs[word]) for word, freq in word_freqs.items()]) # might crash idfs[word] if word not in dict
                heappushpop(heap, (tfidf, doc[0])) if len(heap) > doc_num else heappush(heap, (tfidf, doc[0]))
          
            heap_set = set(heap)
            for did in relations[query_id]:
                if did not in heap_set:
                    heappushpop(heap, (1, did))
            
            out_str = "%d %s %s\n"
            for _, did in heap:
                i = 1 if did in relations[query_id] else 0
                f.write(out_str % (i, query_id, did))
                
        f.close()
    


                            
        # convert heap to set
        # check orig 10 in set
        # sort heap and remove bottom k and add the k missing matches
        # shuffle array ? is this necessary
        # write relations



