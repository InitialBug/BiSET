from collections import Counter
import re
from tqdm import tqdm

from nltk.stem.porter import PorterStemmer
import numpy as np, scipy.stats as st

stemmer = PorterStemmer()


class Rouge(object):
    def __init__(self, stem=True, use_ngram_buf=False):
        self.N = 2
        self.stem = stem
        self.use_ngram_buf = use_ngram_buf
        self.ngram_buf = {}

    @staticmethod
    def _format_sentence(sentence):
        s = sentence.lower()
        s = re.sub(r"[^0-9a-z]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        return s

    def _create_n_gram(self, raw_sentence, n, stem):
        if self.use_ngram_buf:
            if raw_sentence in self.ngram_buf:
                return self.ngram_buf[raw_sentence]
        res = {}
        sentence = Rouge._format_sentence(raw_sentence)
        tokens = sentence.split(' ')
        if stem:
            # try:
            tokens = [stemmer.stem(t) for t in tokens]
            # except:
            #     pass
        sent_len = len(tokens)
        for _n in range(n):
            buf = Counter()
            for idx, token in enumerate(tokens):
                if idx + _n >= sent_len:
                    break
                ngram = ' '.join(tokens[idx: idx + _n + 1])
                buf[ngram] += 1
            res[_n] = buf
        if self.use_ngram_buf:
            self.ngram_buf[raw_sentence] = res
        return res

    def get_ngram(self, sents, N, stem):
        if isinstance(sents, list):
            res = {}
            for _n in range(N):
                res[_n] = Counter()
            for sent in sents:
                ngrams = self._create_n_gram(sent, N, stem)
                for this_n, counter in ngrams.items():
                    # res[this_n] = res[this_n] + counter
                    self_counter = res[this_n]
                    for elem, count in counter.items():
                        if elem not in self_counter:
                            self_counter[elem] = count
                        else:
                            self_counter[elem] += count
            return res
        elif isinstance(sents, str):
            return self._create_n_gram(sents, N, stem)
        else:
            raise ValueError

    def find_lcseque(self,s1, s2):
        # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
        m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
        # d用来记录转移方向
        d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

        for p1 in range(len(s1)):
            for p2 in range(len(s2)):
                if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                    m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                    d[p1 + 1][p2 + 1] = 'ok'
                elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                    m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                    d[p1 + 1][p2 + 1] = 'left'
                else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                    m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                    d[p1 + 1][p2 + 1] = 'up'
        (p1, p2) = (len(s1), len(s2))
        s = []
        while m[p1][p2]:  # 不为None时
            c = d[p1][p2]
            if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
                s.append(s1[p1 - 1])
                p1 -= 1
                p2 -= 1
            if c == 'left':  # 根据标记，向左找下一个
                p2 -= 1
            if c == 'up':  # 根据标记，向上找下一个
                p1 -= 1
        s.reverse()
        return ' '.join(s)

    def get_mean_sd_internal(self, x):
        mean = np.mean(x)
        sd = st.sem(x)
        res = st.t.interval(0.95, len(x) - 1, loc=mean, scale=sd)
        return (mean, sd, res)

    def compute_rouge(self, references, systems):
        assert (len(references) == len(systems))

        # peer_count = len(references)

        result_buf = {}
        for n in range(self.N):
            result_buf[n] = {'p': [], 'r': [], 'f': []}
        result_buf['L'] = {'p': [], 'r': [], 'f': []}
        for ref_sents, sys_sent in zip(references, systems):

            # sys_sent=" ".join(sys_sent.split()[:18])
            # ref_sent = " ".join(ref_sent.split()[:18])
            # if "unk unk" in sys_sent:
            #     print()
            # if len(sys_sent.split())>1:
            #     sys_sent=sys_sent.split()
            #     newstr=[]
            #     tmp=sys_sent[0]
            #     newstr.append(tmp)
            #     for i in range(1,len(sys_sent)):
            #         if tmp!=sys_sent[i]:
            #             newstr.append(sys_sent[i])
            #         tmp=sys_sent[i]
            #     sys_sent=" ".join(newstr)
            match=[0,0]
            alls=[0,0]
            allr=[0,0]
            for ref_sent in ref_sents:
                #rouge-n
                # sys_sent = sys_sent.replace('unknown_word', 'unk')
                # ref_sent = ref_sent.replace('unknown_word', 'unk')
                ref_ngrams = self.get_ngram(ref_sent, self.N, self.stem)
                sys_ngrams = self.get_ngram(sys_sent, self.N, self.stem)
                for n in range(self.N):
                    ref_ngram = ref_ngrams[n]
                    sys_ngram = sys_ngrams[n]
                    ref_count = sum(ref_ngram.values())
                    sys_count = sum(sys_ngram.values())
                    match_count = 0
                    for k, v in sys_ngram.items():
                        if k in ref_ngram:
                            match_count += min(v, ref_ngram[k])
                    match[n]+=match_count
                    alls[n]+=sys_count
                    allr[n]+=ref_count
                    # pn[n]=match_count/sys_count
                    # rn[n] = match_count / ref_count
                    # pn[n] = match_count / sys_count
                    # fn[n] =  0 if (p == 0 or r == 0) else 2 * p * r / (p + r)
            for n in range(self.N):
                p = match[n] / alls[n] if alls[n] != 0 else 0
                r = match[n] / allr[n] if allr[n] != 0 else 0
                f = 0 if (p == 0 or r == 0) else 2 * p * r / (p + r)
                result_buf[n]['p'].append(p)
                result_buf[n]['r'].append(r)
                result_buf[n]['f'].append(f)




        res = {}
        for n in range(self.N):
            n_key = 'rouge-{0}'.format(n + 1)
            res[n_key] = {}
            if len(result_buf[n]['f']) >= 50:
                res[n_key]['p'] = self.get_mean_sd_internal(result_buf[n]['p'])
                res[n_key]['r'] = self.get_mean_sd_internal(result_buf[n]['r'])
                res[n_key]['f'] = self.get_mean_sd_internal(result_buf[n]['f'])
            else:
                # not enough samples to calculate confidence interval
                res[n_key]['p'] = (np.mean(np.array(result_buf[n]['p'])), 0, (0, 0))
                res[n_key]['r'] = (np.mean(np.array(result_buf[n]['r'])), 0, (0, 0))
                res[n_key]['f'] = (np.mean(np.array(result_buf[n]['f'])), 0, (0, 0))

        '''
            alllcs=0
            alls=0
            allr=0
            for ref_sent in ref_sents:
                sys_sent = sys_sent.replace('unknown_word', 'unk')
                ref_sent = ref_sent.replace('unknown_word', 'unk')
                ref_sent_token = Rouge._format_sentence(ref_sent).split()
                sys_sent_token = Rouge._format_sentence(sys_sent).split()
                if self.stem:
                    ref_sent_token = [stemmer.stem(t) for t in ref_sent_token]
                    sys_sent_token = [stemmer.stem(t) for t in sys_sent_token]
                lcs=self.find_lcseque(ref_sent_token,sys_sent_token)
                alllcs+=len(lcs.split())
                alls+=len(sys_sent_token)
                allr+=len(ref_sent_token)
            p=alllcs/alls
            r=alllcs/allr
            f = 0 if (p == 0 or r == 0) else 2 * p * r / (p + r)

            result_buf['L']['p'].append(p)
            result_buf['L']['r'].append(r)
            result_buf['L']['f'].append(f)
            
        n_key = 'rouge-L'
        res[n_key] = {}
        if len(result_buf['L']['f']) >= 50:
            res[n_key]['p'] = self.get_mean_sd_internal(result_buf['L']['p'])
            res[n_key]['r'] = self.get_mean_sd_internal(result_buf['L']['r'])
            res[n_key]['f'] = self.get_mean_sd_internal(result_buf['L']['f'])
        else:
            # not enough samples to calculate confidence interval
            res[n_key]['p'] = (np.mean(np.array(result_buf['L']['p'])), 0, (0, 0))
            res[n_key]['r'] = (np.mean(np.array(result_buf['L']['r'])), 0, (0, 0))
            res[n_key]['f'] = (np.mean(np.array(result_buf['L']['f'])), 0, (0, 0))

        '''
        return res
