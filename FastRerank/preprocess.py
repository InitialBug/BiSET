from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def process_file(config,source,target,template_file,score_file, data_type, word_counter, char_counter,title):

    print("Generating {} examples...".format(data_type))
    examples = []
    with open(source, "r") as fh:
        article=fh.readlines()
    with open(target,'r') as fh:
        template=fh.readlines()
    # with open(template_file,'r') as fh:
    #     template_index=json.load(fh)
    with open(score_file,'r') as fh:
        samples=json.load(fh)
    print(len(samples))

    with open(title,'r') as fh:
        title=fh.readlines()

    def filter_func(example):
        if data_type=='train':
            return len(example) > config.train_para_len
        else:
            return False


    for sample in tqdm(samples):
        if data_type=='train' and len(list(filter(lambda x:x<1.0,sample['scores'])))<config.template_num:
            continue

        art_idx=int(sample['art_idx'])
        article=article[art_idx]
        article = article.replace(
            "''", '" ').replace("``", '" ')

        article_tokens = word_tokenize(article)
        if filter_func(article_tokens):
            continue
        for token in article_tokens:
            word_counter[token] += 1
            # for char in token:
            #     char_counter[char] += 1

        title_tokens=title[art_idx]
        title_tokens = title_tokens.replace(
            "''", '" ').replace("``", '" ')

        title_tokens = word_tokenize(title_tokens)
        k=0
        for index,score in zip(sample['tp_idx'],sample['scores']):
            if score<1.0 or data_type!='train':
                k+=1
                tp=template[index]
                tp = tp.replace(
                    "''", '" ').replace("``", '" ')

                tp_tokens = word_tokenize(tp)
                for token in tp_tokens:
                    word_counter[token] += 1
                    # for char in token:
                    #     char_counter[char] += 1

                example = {"article_tokens": article_tokens,
                           "template_tokens": tp_tokens,
                           "title_tokens":title_tokens,
                            "id": index,"article_id":art_idx,"score":score}
                examples.append(example)
            if k==config.template_num:break
    print("{} samples in total".format(len(examples)))

    return examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    # filtered_elements = [k for k, v in counter.items() if v < limit]
    # print('{} words have been filtered'.format(len(filtered_elements)))
    emb_file=None #you can change this file to GloVe embeddings
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(counter), data_type))


    for word in counter.keys():
        if word not in embedding_dict:
            embedding_dict[word]=np.random.random(size=vec_size)
    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                                     token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    idx2token={idx:token for token,idx in enumerate(token2idx_dict.keys(), 2)}
    return emb_mat, token2idx_dict,idx2token

def build_features(config, examples, data_type, out_file,word2idx_dict, char2idx_dict, is_test=False):
    para_limit = config.dev_para_len if is_test else config.train_para_len
    template_limit=config.template_len
    word_len = config.word_len

    def filter_func(example):
        if is_test:
            return False
        return len(example["article_tokens"]) > para_limit or \
               len(example["template_tokens"]) > template_limit

    print("Processing {} examples...".format(data_type))
    total = 0
    meta = {}
    N = len(examples)
    example_ids=[]
    for n, example in tqdm(enumerate(examples)):
        new_example = {'article_tokens':[],'template_tokens':[]}
        #
        # if filter_func(example):
        #     continue

        total+=1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        # def _get_char(char):
        #     if char in char2idx_dict:
        #         return char2idx_dict[char]
        #     return 1

        for token in example["article_tokens"]:
            new_example['article_tokens'].append(_get_word(token))

        for token in example["template_tokens"]:
            new_example['template_tokens'].append(_get_word(token))

        # for token in example["article_chars"]:
        #     chars=[]
        #     for j, char in enumerate(token):
        #         if j == word_len:
        #             break
        #         chars.append(_get_char(char))
        #     chars+=[0]*(word_len-len(token))
        #     new_example['article_chars'].append(chars)
        #
        # for token in example["template_chars"]:
        #     chars=[]
        #     for j, char in enumerate(token):
        #         if j == word_len:
        #             break
        #         chars.append(_get_char(char))
        #     chars+=[0]*(word_len-len(token))
        #     new_example['template_chars'].append(chars)

        if len(new_example['article_tokens'])>para_limit:
            new_example['article_tokens']=new_example['article_tokens'][:para_limit]
        if len(new_example['template_tokens'])>template_limit:
            new_example['template_tokens']=new_example['template_tokens'][:template_limit]

        # new_example['score']=rouge_n(example['title_tokens'],example['template_tokens'],1)
        new_example['score']=example['score']

        new_example['id']=example["id"]
        new_example['article_id']=example['article_id']
        example_ids.append(new_example)

    save(out_file,example_ids,message=out_file)
    print("Built {} / {} instances of features in total".format(total, N))
    meta["total"] = total
    return meta

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def preproc(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples = process_file(config,
        config.train_article, config.train_title,config.train_template_index,config.train_scores,"train", word_counter, char_counter,config.train_title)
    dev_examples= process_file(config,
        config.dev_article,config.train_title,config.dev_template_index, config.dev_scores,"dev", word_counter, char_counter,config.dev_title)
    test_examples= process_file(config,
        config.test_article,config.train_title,config.test_template_index, config.test_scores,"test", word_counter, char_counter,config.test_title)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict,idx2token= get_embedding(
        word_counter, "word", emb_file=word_emb_file, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict ,idx2token= get_embedding(
        char_counter, "char", emb_file=char_emb_file, vec_size=char_emb_dim)

    build_features(config, train_examples, "train",
                   config.train_token_file,word2idx_dict, None)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_token_file, word2idx_dict, None,is_test=True)
    with open(config.word2idx_file,'r') as fh:
        word2idx_dict=json.load(fh)
    build_features(config, test_examples, "test",
                   config.test_token_file, word2idx_dict, None, is_test=True)


    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.word2idx_file, word2idx_dict, message="word dictionary")
    save(config.idx2token_file,idx2token,message="idx2token")