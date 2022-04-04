# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/attention.py
# NYC: NY commented
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

# # Additive Attention From Tacotron 1
class BahdanauAttention(nn.Module):
    def __init__(self, dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.tanh == nn.Tanh()
        self.v = nn.Linear(dim, 1, bias=False)
    
    def forward(self, query, processed_memory):
        # ''' Copilot 자동생성 구문. 
        # processed_query = self.query_layer(query)
        # score = self.v(self.tanh(processed_query + processed_memory))
        # align = F.softmax(score, dim=1)
        # context = align.bmm(processed_memory)
        # return context, align
        # '''
        # # query : (batch, 1, dim) or (batch, dim)
        # #         output of RNN cell (cell_output)
        # # processed_memory : (batch, max_time, dim) # NYC: Tacotron1 은 padding이 있었나? 아무튼 고정 길이였다고.

        if query.dim() == 2: # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        
        processed_query = self.query_layer(query)
        alignment = self.v(self.tanh(processed_query + processed_memory))

        # (batch, max_time)
        return alignment.squeeze(-1)

def get_mask_from_lengths(memory, memory_lengths):
    # # memory_lengths : (batch, max_time)
    # # memory_lengths : array like (NYC: (batch)?)
    # # mask : (batch, max_time, 1) (NYC: copilot wrote)
    
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask 
    # '''
    # mask = memory.new_ones(memory.size(0), memory.size(1))
    # for i, l in enumerate(memory_lengths): mask[i, l:] = 0
    # return mask
    # '''

class AttentionWrapper(nn.Module):
    def __init__(self, rnn_cell, attention_mechanism, score_mask_value=-float("inf")):
        super(AttentionWrapper, self).__init__()
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism
        self.score_mask_value = score_mask_value
    
    def forward(self, query, attention, cell_state, memory, 
                processed_memory=None, mask=None, memory_lengths=None):
        # ''' Copilot 자동생성 구문. 
        # if processed_memory is None:
        #     processed_memory = self.attention_mechanism(memory)
        # if mask is None:
        #     mask = get_mask_from_lengths(memory, memory_lengths)
        # score = self.attention_mechanism(query, processed_memory, mask=mask)
        # align = F.softmax(score, dim=1)
        # context = align.bmm(memory)
        # cell_state = self.rnn_cell(query, context, cell_state)
        # return cell_state, align
        # '''
        # # query : (batch, 1, dim) 
        # # attention : (batch, max_time) ??????
        # # cell_state : (batch, dim)
        # # memory : (batch, max_time, dim)
        # # processed_memory : (batch, max_time, dim)
        # # mask : (batch, max_time, 1) ???????
        # # memory_lengths : (batch)

        if processed_memory is None:
            processed_memory = memory
        if memory_lengths is not None and mask is None:
            mask = get_mask_from_lengths(memory, memory_lengths)

        # Concat input query and previout attention context
        # NYC: 여기의 query와 bahdanau_attention의 query는 다를 듯? 코파일럿은 다르대ㅎ
        # NYC: 여기의 attention 은 무엇인가..?? 그리고 아래 bmm은??
        cell_input = torch.cat((query, attention), -1) 
        # Feed it to RNN
        cell_output = self.rnn_cell(cell_input, cell_state)
        
        # Alignment (batch, max_time)
        alignment = self.attention_mechanism(cell_output, processed_memory)
        if mask is not None:
            mask = mask.view(query.size(0), -1) # dimension 에 맞도록..? dim 재조정 
            alignment.data.masked_fill_(mask, self.score_mask_value)
        
        # Normalize attention weight
        alignment = F.softmax(alignment)  # '생성 매 스텝에서 어느 철자에 제일 집중해야 하는지'
        # Attention context vector (batch, 1, dim)
        attention = torch.bmm(alignment.unsqueeze(1), memory) # memory랑 계산하려고 잠깐.. 하나봐
        attention = attention.squeeze(1) # (batch, dim)

        # NYC: attention은 뭐고, alignment는 무엇이냐? 
        return cell_output, attention, alignment



        
        score = self.attention_mechanism(query, processed_memory, mask=mask)
        align = F.softmax(score, dim=1)
        context = align.bmm(memory)
        cell_state = self.rnn_cell(query, context, cell_state)
        return cell_state, align