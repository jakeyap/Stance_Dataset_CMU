#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:58:11 2020

@author: jakeyap
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from transformers.modeling_bert import BertPreTrainedModel, BertPooler
from transformers import AutoModel
from torch.nn import MSELoss, CrossEntropyLoss, Dropout

categories = {'Explicit_Denial':0,
              'Implicit_Denial':0,
              'Implicit_Support':0,
              'Explicit_Support':0,
              'Comment':0,
              'Queries':0}

    

class my_ModelA0(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(my_ModelA0, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout0 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier0 = torch.nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        
        output = outputs[1]
        pooled_output = self.dropout0(output)
        logits = self.classifier0(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    
class my_ModelB0(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(my_ModelB0, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout1 = Dropout(config.hidden_dropout_prob)
        # The plus 1 factor is to stitch in the input label
        self.classifier1 = torch.nn.Linear(config.hidden_size+1, config.hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.dropout2 = Dropout(config.hidden_dropout_prob)
        self.classifier2 = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, 
                interaction=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        
        output = outputs[1]
        output = self.dropout1(output)
        output = self.classifier1(torch.cat((output, interaction), 1))
        output = self.relu1(output)
        output = self.dropout2(output)
        logits = self.classifier2(output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class my_ModelE0(BertPreTrainedModel):
    """
    This model has 2 classifiers. 
    classifier0 is for a pretraining task to learn twitter-speak. Check whether words have been swapped
    classifier1 is for the actual task training of learning stance
    
    Examples::
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        
    """
    def __init__(self, config):
        super(my_ModelE0, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout0 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier0 = torch.nn.Linear(config.hidden_size, 2) # for swapped words
        self.classifier1 = torch.nn.Linear(config.hidden_size, 4) # for stance
        
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, task='stance'):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)
        
        output = outputs[1] # get only the last layer outputs
        pooled_output = self.dropout0(output)
        if task=='stance':
            logits = self.classifier1(pooled_output)
            # shape of outputs[0] is (n,4)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        elif task=='pretrain':
            logits = self.classifier0(pooled_output)
            # shape of outputs[0] is (n,1)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        else:
            raise Exception('task not found : ' + task)

        return outputs  # logits, (hidden_states), (attentions)

class my_Bertweet(nn.Module):       # used in main_multitask.py
    def __init__(self, num_labels, dropout):
        super(my_Bertweet, self).__init__()
        self.num_labels = num_labels

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        config = self.bertweet.config
        self.dropout0 = torch.nn.Dropout(dropout)
        self.classifier0 = torch.nn.Linear(config.hidden_size, 4) # for stance
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None):
        
        outputs = self.bertweet(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids)
        
        output = outputs[1] # get only the last layer outputs
        pooled_output = self.dropout0(output)
        
        logits = self.classifier0(pooled_output)
        # shape of outputs[0] is (n,4)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)
    
class mtt_Bertweet(nn.Module):      # used in main_multitask.py
    ''' For multitask. Viral and stance prediction '''
    def __init__(self, num_labels, dropout):
        super(mtt_Bertweet, self).__init__()
        self.num_labels = num_labels

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        config = self.bertweet.config
        self.dropout0 = torch.nn.Dropout(dropout)
        self.classifier0 = torch.nn.Linear(config.hidden_size, 4) # for stance
        self.classifier1 = torch.nn.Linear(config.hidden_size, 2) # for viral
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, task='stance'):
        
        outputs = self.bertweet(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids)
        
        output = outputs[1] # get only the last layer outputs
        pooled_output = self.dropout0(output)
        
        stance_logits = None
        viral_logits = None
        if task not in ['stance','multi','viral']:
            raise Exception('Task not found: ' + task)
        if task in ['stance','multi']:
            stance_logits = self.classifier0(pooled_output)
        if task in ['viral','multi']:
            viral_logits = self.classifier1(pooled_output)
        
        # shape of stance_logits is (n,4), viral_logits is (n,2)
        outputs = (stance_logits, viral_logits, ) + outputs[2:]  # add hidden states and attention if they are here
        
        return outputs

class mtt_Bertweet2(nn.Module):    # used in main_multitask_user_features.py
    ''' 
    For multitask. Viral and stance prediction. Includes the other meta data features
    For stance task, head and tail tweets are encoded by bertweet respectively. The outputs are passed into a transformer
    
    '''
    def __init__(self, num_labels, dropout, num_layers=2):
        super(mtt_Bertweet2, self).__init__()
        self.num_labels = num_labels

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        config = self.bertweet.config
        self.dropout0 = torch.nn.Dropout(dropout)
        self.classifier0 = torch.nn.Linear(config.hidden_size * 2 + 3, 4) # for stance
        self.classifier1 = torch.nn.Linear(config.hidden_size * 2 + 3, 2) # for viral
        
        # a single self attention layer
        single_tf_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size * 2, 
                                                     nhead=8, 
                                                     dropout=config.hidden_dropout_prob)
        # create transformer blocks. the single layer is cloned internally
        self.transformer_block = nn.TransformerEncoder(single_tf_layer,
                                                       num_layers=num_layers)
        
    def forward(self, 
                input_ids_h, attention_mask_h=None, token_type_ids_h=None, position_ids_h=None, 
                input_ids_t=None, attention_mask_t=None, token_type_ids_t=None, position_ids_t=None, 
                followers_head=None, followers_tail=None, int_type_num=None,
                task='stance'):
        
        outputs_h = self.bertweet(input_ids_h,
                                  attention_mask=attention_mask_h,
                                  token_type_ids=token_type_ids_h,
                                  position_ids=position_ids_h)
        
        outputs_t = self.bertweet(input_ids_t,
                                  attention_mask=attention_mask_t,
                                  token_type_ids=token_type_ids_t,
                                  position_ids=position_ids_t)
        
        output_h = outputs_h[1]                         # get the last layer outputs. shape=(n,768)
        output_t = outputs_t[1]                         # get the last layer outputs. shape=(n,768)
        pooled_output_h = self.dropout0(output_h)       # apply dropout. shape=(n,768)
        pooled_output_t = self.dropout0(output_t)       # apply dropout. shape=(n,768)
        pooled_output   = torch.cat((pooled_output_h,   # concat pooled outputs. shape=(n,1536)
                                     pooled_output_t), 
                                    dim=1)          
        
        pooled_output = pooled_output.unsqueeze(1)      # shape=(n,1,1536)
        tmp = self.transformer_block(pooled_output)     # apply transformer. shape=(n,1,1536)
        tmp = tmp.squeeze(1)                            # shape=(n,1536)
        metadata = torch.cat((followers_head,           # each shape=(n,1)
                              followers_tail,           # after cat, shape=(n,3)
                              int_type_num),
                             dim=1)
        
        tmp = torch.cat((tmp, metadata), dim=1)         # shapes were (n,1536) and (n,3). now (n,1539)
        tmp = self.dropout0(tmp)                        # apply dropout again. shape=(n,1539)
        
        stance_logits = None
        viral_logits = None
        if task not in ['stance','multi','viral']:
            raise Exception('Task not found: ' + task)
        if task in ['stance','multi']:
            stance_logits = self.classifier0(tmp)
        if task in ['viral','multi']:
            viral_logits = self.classifier1(tmp)
        
        # shape of stance_logits is (n,4), viral_logits is (n,2)
        outputs = (stance_logits, viral_logits, ) + outputs_h[2:] + outputs_t[2:]   # add hidden states and attention if they are here
        
        return outputs


class SelfAdjDiceLoss(torch.nn.Module):
    """
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")