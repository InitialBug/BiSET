import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#--------------------------------------------------Base Module--------------------------------------------------#

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size, conv_dim=1, padding=0, bias=True, activation=False,dilation=1):
        super().__init__()
        self.activation = activation
        if conv_dim == 1:
            self.conv = nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size,
                                            padding=padding, bias=bias,dilation=dilation)

        if activation:
            nn.init.kaiming_normal_(self.conv.weight,nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.conv.weight)

    def forward(self, input):
        input = input.transpose(1, 2)
        out = self.conv(input)
        out = out.transpose(1, 2)
        if self.activation:
            return F.relu(out)
        else:
            return out

class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x

def GLU(input):
    out_dim=input.shape[2]//2
    a,b=torch.split(input,out_dim,dim=2)
    return a*F.sigmoid(b)

class DimReduce(nn.Module):
    def __init__(self, input_dim, out_dim,kernel_size):
        super().__init__()
        self.convout = nn.Conv1d(input_dim, out_dim*2, kernel_size, padding=kernel_size // 2)
        nn.init.xavier_normal_(self.convout.weight)

    def forward(self, input):
        input = input.transpose(1, 2)
        input = self.convout(input)
        input = input.transpose(1, 2)
        out=GLU(input)
        return out

class Embedding(nn.Module):
    def __init__(self, char_dim, word_dim, word_len, out_dim, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1d = DepthwiseSeparableConv(char_dim, char_dim, kernel_size, padding=kernel_size // 2,activation=True)
        self.highway = Highway(2, char_dim + word_dim)
        self.dropout = dropout
        self.char_dim = char_dim
        self.word_len = word_len

    def forward(self, word_emb, char_emb):
        batch_size = word_emb.shape[0]
        seq_len = word_emb.shape[1]
        char_emb = char_emb.view([-1, self.word_len, self.char_dim])
        char_emb = self.conv1d(char_emb)
        char_emb, _ = torch.max(char_emb, dim=1)
        char_emb = char_emb.view(batch_size, seq_len, self.char_dim)
        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.highway(emb)
        return emb



class Similarity(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.ff1=nn.Linear(10,10)
        self.ff2=nn.Linear(10,1)

    def forward(self, article, template,article_mask,template_mask):
        article_len = article.shape[1]
        template_len = template.shape[1]
        c=article.unsqueeze(dim=2)
        c=c.repeat([1,1,template_len,1])
        q=template.unsqueeze(dim=1)
        q=q.repeat([1,article_len,1,1])

        S=q-c
        S=torch.sum(S*S,dim=3)

        S=torch.exp(-1*S)
        article_mask=article_mask.unsqueeze(dim=2)
        article_mask=article_mask.repeat([1,1,template_len])
        template_mask=template_mask.unsqueeze(dim=1)
        template_mask=template_mask.repeat([1,article_len,1])
        S=S*article_mask*template_mask

        row_max,_=torch.max(S,dim=2)
        row_max,_=row_max.topk(10,dim=1,largest=True)

        row_max=F.relu(self.ff1(row_max))
        row_max=self.ff2(row_max)
        out=row_max
        out=out.squeeze()

        return out

#--------------------------------------------------GLDR Module--------------------------------------------------#

class ResidualBlock(nn.Module):
    def __init__(self, input_dim,kernel_size,dilation):
        super().__init__()
        self.conv1=DepthwiseSeparableConv(input_dim,input_dim*2,kernel_size,padding=kernel_size//2*dilation,dilation=dilation)

    def forward(self, input):
        out=self.conv1(input)
        out=GLU(out)
        out=input+out
        return out

class ConvEncoder(nn.Module):
    def __init__(self,input_dim,kernel_size,conv_num,exp_num=5,refine_num=3,dropout=0.1):
        super().__init__()
        self.dropout=dropout
        self.exp_conv=nn.Sequential()
        dilation=1
        for i in range(conv_num):
            self.exp_conv.add_module(str(i),ResidualBlock(input_dim,kernel_size,dilation))
            if i<exp_num:
                dilation*=2
        self.refine=nn.Sequential()
        for i in range(refine_num):
            self.refine.add_module(str(i),ResidualBlock(input_dim,kernel_size,dilation=1))


    def forward(self, input):
        out=self.exp_conv(input)
        out=self.refine(out)
        out=F.dropout(out,p=self.dropout,training=self.training)
        return out


class FastRerank(nn.Module):
    def __init__(self, char_dim, char_vocab_size, word_len, word_dim, word_mat, emb_dim, kernel_size,
                 encoder_block_num,model_block_num, dropout=0.1):
        super().__init__()

        self.word_emb = nn.Embedding(word_mat.shape[0],word_dim)

        self.conv_encoder=ConvEncoder(word_dim,kernel_size,conv_num=encoder_block_num,dropout=dropout)
        self.similarity = Similarity(word_dim, dropout)


    def forward(self, article_word, article_char, template_word, template_char,article_mask,template_mask):
        article_word = self.word_emb(article_word)
        template_word = self.word_emb(template_word)
        article=article_word
        template=template_word
        article = self.conv_encoder(article)
        template = self.conv_encoder(template)
        score = self.similarity(article, template,article_mask,template_mask)
        return score
