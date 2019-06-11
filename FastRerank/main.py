import os
import numpy as np
import ujson as json
from tqdm import tqdm
import torch
import torch.optim as optim
from model import FastRerank
import torch.nn.functional as F
from PyRouge.Rouge import Rouge
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(message)s')


class Dataset:
    def __init__(self, data_file,config,train=True):
        with open(data_file, "r") as fh:
            self.data = json.load(fh)
        self.data_size = len(self.data)
        self.indices = list(range(self.data_size))
        self.train=train
        self.config=config
        if train:
            self.para_len=config.train_para_len
        else:
            self.para_len=config.dev_para_len

    def gen_batches(self, batch_size, shuffle=True, pad_id=0):
        if shuffle:
            np.random.shuffle(self.indices)
        for batch_start in np.arange(0, self.data_size, batch_size):
            batch_indices = self.indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(batch_indices, pad_id)

    def _one_mini_batch(self, indices, pad_id):
        article_word,article_mask,article_len = self.dynamic_padding('article_tokens', indices, pad_id)
        template_word,template_mask,template_len = self.dynamic_padding('template_tokens', indices, pad_id)
        # article_char = self.dynamic_padding('article_chars', indices, pad_id, ischar=True)
        # template_char = self.dynamic_padding('template_chars', indices, pad_id, ischar=True)
        article_char=torch.Tensor([])
        template_char=torch.Tensor([])
        scores = [self.data[i]['score'] for i in indices]
        ids = [self.data[i]['id'] for i in indices]
        article_id=[self.data[i]['article_id'] for i in indices]

        # article_len=self.get_len(article_len)
        res = (torch.Tensor(article_word).long(), torch.Tensor(article_char).long(), torch.Tensor(template_word).long(),
               torch.Tensor(template_char).long(), torch.Tensor(scores).float(),
               ids,torch.Tensor(article_mask).float(),torch.Tensor(template_mask).float(),article_id)


        return res


    def dynamic_padding(self, key_word, indices, pad_id,max_len=10, ischar=False):
        sample = []
        length=[]
        for i in indices:
            sample.append(self.data[i][key_word])
            l=len(self.data[i][key_word])
            max_len = max(max_len, l)
            length.append(l)
        if ischar:
            pads = [pad_id] * self.config.word_len
            pad_sample = [ids + [pads] * (max_len - len(ids)) for ids in sample]
            return pad_sample
        else:
            pad_sample = [ids + [pad_id] * (max_len - len(ids)) for ids in sample]
            mask=[[1]*len(ids)+[0]*(max_len - len(ids)) for ids in sample]
            return pad_sample,mask,length

    def __len__(self):
        return self.data_size

def getlogger(log_path):
    logger = logging.getLogger("FastRerank")
    fh = logging.FileHandler(log_path, mode='a')
    ch = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

import time

def train(config,device):
    logger = getlogger(config.train_log)
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char2idx_file, "r") as fh:
        char_dict = json.load(fh)

    with open(config.train_title,'r') as fh:
        temp=fh.readlines()
    rouge_calculator = Rouge.Rouge()
    with open(config.dev_title, 'r') as fh:
        dev_title = fh.readlines()

    logger.info("Building model...")

    train_dataset = Dataset(config.train_token_file,config)
    train_it_num = len(train_dataset) // config.batch_size
    dev_dataset = Dataset(config.dev_token_file,config,train=False)

    dev_it_num = len(dev_dataset) // config.val_batch_size

    char_vocab_size = len(char_dict)
    del char_dict
    model = FastRerank(config.char_dim, char_vocab_size, config.word_len, config.glove_dim, word_mat,
                      config.emb_dim, config.kernel_size,config.encoder_block_num,config.model_block_num).to(device)

    if config.model:
        model.load_state_dict(torch.load(os.path.join(config.save_dir, config.model)))

    model.train()
    parameters = filter(lambda param: param.requires_grad, model.parameters())


    optimizer = optim.Adam(weight_decay=config.L2_norm, params=parameters,lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10000,gamma=0.1)


    loss_func=torch.nn.BCEWithLogitsLoss()

    steps = 0
    patience = 0
    losses=0
    min_loss=10000
    start_time=time.time()

    for epoch in range(config.epochs):
        batches = train_dataset.gen_batches(config.batch_size,shuffle=True)
        for batch in tqdm(batches, total=train_it_num):
            optimizer.zero_grad()
            (contex_word, contex_char, template_word, template_char, scores, ids,contex_mask,template_mask,art_id) = batch
            contex_word, contex_char, template_word, template_char = contex_word.to(device), contex_char.to(
                device), template_word.to(device), template_char.to(device)
            contex_mask, template_mask,scores = contex_mask.to(device), template_mask.to(device),scores.to(device)
            p= model(contex_word, contex_char, template_word, template_char,contex_mask, template_mask)

            loss = loss_func(p,scores)
            losses+=loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,config.grad_clip)
            scheduler.step()
            optimizer.step()

            if (steps + 1) % config.checkpoint == 0:

                losses=losses/config.checkpoint
                log_ = 'itration {} train loss {}\n'.format(steps, losses)
                logger.info(log_)
                losses=0
                batches = dev_dataset.gen_batches(config.val_batch_size, shuffle=False)
                template = []
                for batch in tqdm(batches, total=dev_it_num):
                    (contex_word, contex_char, template_word, template_char, scores, ids, contex_mask,
                     template_mask,art_id) = batch
                    contex_word, contex_char, template_word, template_char = contex_word.to(device), contex_char.to(
                        device), template_word.to(device), template_char.to(device)
                    contex_mask, template_mask, scores = contex_mask.to(device), template_mask.to(device), scores.to(
                        device)

                    p = model(contex_word, contex_char, template_word, template_char, contex_mask, template_mask,
                            )
                    loss = loss_func(p, scores)
                    losses += loss.item()
                losses/=dev_it_num
                log_ = 'itration {} dev loss {}\n'.format(steps, losses)
                logger.info(log_)

                if losses<min_loss:
                    patience=0
                    min_loss=losses
                    fn = os.path.join(config.save_dir, "model_{}.pkl".format(min_loss))
                    torch.save(model.state_dict(), fn)
                else:
                    patience+=1
                    if patience>config.early_stop:
                        print('early stop because val loss is continuing incresing!')
                        end_time=time.time()
                        logger.info("total training time{}".format(end_time-start_time))
                        exit()
                losses=0

            steps += 1
    fn = os.path.join(config.save_dir, "model_final.pkl")
    torch.save(model.state_dict(), fn)


def test(config):
    pass


import re
def format_sentence(sentence):
    s = sentence.lower()
    s = re.sub(r"[^0-9a-z]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def dev(config,device):
    keyword='test'
    logger = getlogger(config.dev_log)
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char2idx_file, "r") as fh:
        char_dict = json.load(fh)

    logger.info("Building dev/test model...")
    if keyword=='train':
        dev_dataset = Dataset(config.train_token_file,config,train=False)
        with open(config.train_article,'r') as fh:
            article=fh.readlines()
        with open(config.train_title, 'r') as fh:
            dev_title = fh.readlines()
    elif keyword=='dev':
        dev_dataset = Dataset(config.dev_token_file,config,train=False)
        with open(config.dev_article,'r') as fh:
            article=fh.readlines()
        with open(config.dev_title, 'r') as fh:
            dev_title = fh.readlines()
    else:
        dev_dataset = Dataset(config.test_token_file,config,train=False)
        with open(config.test_article,'r') as fh:
            article=fh.readlines()
        with open(config.test_title, 'r') as fh:
            dev_title = fh.readlines()

    dev_it_num = len(dev_dataset) // config.val_batch_size
    batches = dev_dataset.gen_batches(config.val_batch_size, shuffle=False)
    char_vocab_size = len(char_dict)
    del char_dict
    model = FastRerank(config.char_dim, char_vocab_size, config.word_len, config.glove_dim, word_mat,
                      config.emb_dim, config.kernel_size,config.encoder_block_num,config.model_block_num).to(device)

    if not config.model:
        raise Exception('Empty parameter of --model')
    model.load_state_dict(torch.load(os.path.join(config.save_dir, config.model)))
    model.eval()
    loss_func = torch.nn.BCEWithLogitsLoss()

    losses = 0
    template=[]
    rewrite_sample=[]

    template_txt=open('/data/wangkai/BiSET/data/giga/{}.template.{}'.format(keyword,config.template_num),'w')


    with open(config.train_title,'r') as fh:
        temp=fh.readlines()
    k=0
    for batch in tqdm(batches, total=dev_it_num):
        (contex_word, contex_char, template_word, template_char, scores, ids, contex_mask, template_mask,art_id) = batch
        contex_word, contex_char, template_word, template_char = contex_word.to(device), contex_char.to(
            device), template_word.to(device), template_char.to(device)
        contex_mask, template_mask, scores = contex_mask.to(device), template_mask.to(device), scores.to(device)
        p = model(contex_word, contex_char, template_word, template_char, contex_mask, template_mask)
        loss = loss_func(p, scores)
        losses += loss.item()
        p=p.view(-1,config.template_num)

        _,index=torch.max(p,dim=1)

        for i in range(len(index)):
            idx=index[i] + config.template_num* i
            id = ids[idx]
            template.append(temp[id])
            template_txt.write(temp[id])
            sample={"article":article[art_id[idx]],"title":dev_title[art_id[idx]],"template":temp[id]}
            rewrite_sample.append(sample)

            k+=1
    losses/=k
    log_='loss:{}'.format(losses)
    logger.info(log_)
    if keyword=='train':
        with open(config.train_template, 'w') as fh:
            json.dump(rewrite_sample, fh)
    elif keyword=='dev':
        with open(config.dev_template, 'w') as fh:
            json.dump(rewrite_sample, fh)
    else:
        with open(config.test_template, 'w') as fh:
            json.dump(rewrite_sample, fh)

    template_txt.close()

    p=open('/data/wangkai/FastRerank/data/{}.pred.txt'.format(keyword),'w')
    with open('/data/wangkai/BiSET/data/giga/{}.template.{}'.format(keyword,config.template_num),'r') as fh:
        new=fh.readlines()
        for n in new:
            p.write(format_sentence(n)+'\n')


