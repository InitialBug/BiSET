"""Define RNN-based encoders."""
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
import torch


class T2A(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.W=nn.Linear(dim,dim,bias=False)
        self.U=nn.Linear(dim,dim,bias=False)
        self.b=nn.Parameter(torch.zeros(dim))

    def forward(self, article_hidden,template_hidden):
        seq_len=template_hidden.shape[0]
        article_hidden=article_hidden[-1,:,:].repeat(seq_len,1,1)
        s=self.W(article_hidden)+self.U(template_hidden)+self.b
        s=template_hidden*F.sigmoid(s)
        return s

class A2T(nn.Module):
    '''
    args:
        1,batch,dim
    return:
        scaler
    '''
    def __init__(self,dim,att_type='dot'):
        super().__init__()
        assert att_type in ['dot','general','mlp']
        self.att_type=att_type
        if att_type=='general':
            self.W=nn.Linear(dim,dim,bias=False)
        else:
            self.W=nn.Linear(dim*2,dim,bias=False)
            self.V=nn.Linear(dim,1,bias=False)

    def forward(self, x,y):
        # if self.att_type=='mlp':
        #     z=torch.cat([x,y],dim=2)
        #     z=F.tanh(self.W(z))
        #     z=self.V(z)
        #     return F.sigmoid(z)
        x=x.transpose(0,1)
        y=y.permute(1,2,0)
        if self.att_type=='general':
            x = self.W(x)
        return F.sigmoid(torch.bmm(x,y))

class BiSET(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.T2A=T2A(hidden_size*2)
        self.A2T=A2T(hidden_size*2,att_type='general')

    def forward(self, article_hidden,template_hidden):
        s = self.A2T(template_hidden[-1:, :], article_hidden[-1:, :])
        s = s.repeat(1, article_hidden.shape[0], 500)
        s = s.transpose(0, 1)

        gate_memory_bank = self.T2A(template_hidden, article_hidden)
        memory_bank = (1 - s) * article_hidden + s * gate_memory_bank
        return memory_bank

class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

        self.no_pack_padded_seq=True
        self.BiSET=BiSET(hidden_size)
    def forward(self, src, template,lengths=None,tp_lengths=None):

        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb)
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)


        #--------------------template encoder-------------------------------------#
        self._check_args(template, tp_lengths)

        emb = self.embeddings(template)

        packed_emb = emb
        if tp_lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            tp_lengths = tp_lengths.view(-1).tolist()
            packed_emb = pack(emb, tp_lengths)

        tp_memory_bank, tp_encoder_final = self.rnn(packed_emb)
        if tp_lengths is not None and not self.no_pack_padded_seq:
            tp_memory_bank = unpack(tp_memory_bank)[0]

        memory_bank=self.BiSET(memory_bank,tp_memory_bank)

        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
