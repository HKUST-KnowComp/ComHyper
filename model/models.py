import math
import torch
import copy
import torch.nn as nn 
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

def linear_block(input_dim, hidden_dim):

    linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.5))

    return linear

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers 
        self.hidden_size = hidden_dim
        
        layers = []
        for i in range(num_layers-1):
            layers.extend(
                linear_block(hidden_dim if i> 0 else input_dim, hidden_dim)
            )
        layers.extend([nn.Linear(hidden_dim, input_dim)])

        self.model = nn.Sequential(*layers)

        ## initilize the model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in,_ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1/math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self,x):
        out = self.model(x)
        return out


class SDSN(nn.Module):
    """docstring for SDSNA"""
    # Replace simple dot product with SDSNA
    # Scoring Lexical Entailment with a supervised directional similarity network
    def __init__(self, arg):
        super(SDSNA, self).__init__()
        
        self.emb_dim = 300
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.map_linear_left = self.mlp(self.emb_dim, self.hidden_dim, self.num_layers)
        self.map_linear_right = self.mlp(self.emb_dim, self.hidden_dim, self.num_layers)

        self.final_linear = nn.Linear(2 * self.hidden_dim + self.emb_dim, 1)

    def init_embs(self, w2v_weight):
        self.embs = nn.Embedding.from_pretrained(w2v_weight, freeze=True)

    def forward(self, inputs):

        batch_size, _ = inputs.size()
        left_w2v = self.embs(inputs[:,0])
        right_w2v = self.embs(inputs[:,1])

        left_trans = self.map_linear_left(left_w2v)
        right_trans = self.map_linear_right(right_w2v)

    def mlp(self, input_dim, hidden_dim, num_layers):
        layers = []
        for i in range(num_layers-1):
            layers.extend(
                linear_block(hidden_dim if i> 0 else input_dim, hidden_dim)
            )
        layers.extend([nn.Linear(hidden_dim, input_dim)])

        return nn.Sequential(*layers)


class Word2Score(nn.Module):
    """docstring for Word2Score"""
    def __init__(self, hidden_dim, num_layers):
        super(Word2Score, self).__init__()
        
        self.emb_dim = 300
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.map_linear_left = self.mlp(self.emb_dim, self.hidden_dim, self.num_layers)
        self.map_linear_right = self.mlp(self.emb_dim, self.hidden_dim, self.num_layers)

    def init_emb(self, w2v_weight):
        self.embs = nn.Embedding.from_pretrained(w2v_weight, freeze=True)

    def mlp(self, input_dim, hidden_dim, num_layers):
        layers = []
        for i in range(num_layers-1):
            layers.extend(
                linear_block(hidden_dim if i> 0 else input_dim, hidden_dim)
            )
        layers.extend([nn.Linear(hidden_dim, input_dim)])

        return nn.Sequential(*layers)

    def forward(self, inputs):

        # inputs: [batch_size, 2]
        batch_size, _ = inputs.size()
        left_w2v  = self.embs(inputs[:,0])
        right_w2v = self.embs(inputs[:,1])

        left_trans = self.map_linear_left(left_w2v)
        right_trans = self.map_linear_right(right_w2v)

        output = torch.einsum('ij,ij->i', [left_trans, right_trans])

        left_norm = torch.norm(left_trans, dim=1).sum()
        right_norm = torch.norm(right_trans, dim=1).sum()

        return output, (left_norm+right_norm)
        
    def inference(self, left_w2v, right_w2v):

        left_trans = self.map_linear_left(left_w2v)
        right_trans = self.map_linear_right(right_w2v)

        output = torch.einsum('ij,ij->i', [left_trans, right_trans])

        return output

class MEAN_Max(nn.Module):
    """docstring for MEAN"""
    def __init__(self, input_dim, hidden_dim):
        super(MEAN_Max, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(0)
        self.output_layer = nn.Sequential( 
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, input_dim)
                        )

    def forward(self, embed_input_left, embed_input_right):
        # input: [batch, context, seq, emb]
        batch_size, num_context, seqlen, emb_dim = embed_input_left.size()
        
        # [batch, context, seq, emb]
        embed_input_left = self.dropout_layer(embed_input_left)
        embed_input_right = self.dropout_layer(embed_input_right)

        oe = torch.cat((embed_input_left, embed_input_right), 2)
        oe = oe.mean(2)
        oe = self.output_layer(oe)
        oe = oe.max(1)[0]
        return oe
 

class MEAN(nn.Module):
    """docstring for MEAN"""
    def __init__(self, input_dim, hidden_dim):
        super(MEAN, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(0)
        self.output_layer = nn.Sequential( 
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, input_dim)
                        )

    def forward(self, embed_input_left, embed_input_right):
        # input: [batch, context, seq, emb]
        batch_size, num_context, seqlen, emb_dim = embed_input_left.size()
        
        # [batch, context, seq, emb]
        embed_input_left = self.dropout_layer(embed_input_left)
        embed_input_right = self.dropout_layer(embed_input_right)

        oe = torch.cat((embed_input_left, embed_input_right), 2)
        oe = oe.mean(2)
        oe = self.output_layer(oe)
        oe = oe.mean(1)
        return oe

class LSTM(nn.Module):
    """docstring for LSTM"""
    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(p=0)
        self.left_context_encoder = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.right_context_encoder = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.output_layer = nn.Sequential( 
                            nn.Linear(hidden_dim*2, hidden_dim*2),
                            nn.ReLU(),
                            nn.Linear(hidden_dim*2, input_dim)
                        )

    def forward(self, embed_input_left, embed_input_right):
        # input: [batch, context, seq, emb]
        batch_size, num_context, seqlen, emb_dim = embed_input_left.size()
  
        # [batch, context, seq, dim]
        embed_input_left = embed_input_left.view(-1, seqlen, self.input_dim)
        embed_input_left = self.dropout_layer(embed_input_left)

        embed_input_right = embed_input_right.view(-1, seqlen, self.input_dim)
        embed_input_right = self.dropout_layer(embed_input_right)

        # hidden = (torch.zeros(1, batch_size*num_context, self.hidden_dim),
        #           torch.zeros(1, batch_size*num_context, self.hidden_dim))

        output_left, (final_hidden_state_left, final_cell_state_left) = self.left_context_encoder(embed_input_left) #, hidden)
        output_right,(final_hidden_state_right, final_cell_state_left) = self.right_context_encoder(embed_input_right) #, hidden)

        encode_context_left = final_hidden_state_left.view(-1, num_context, self.hidden_dim)
        encode_context_right = final_hidden_state_right.view(-1, num_context, self.hidden_dim)

        # concat + mean_pooling + fully_connect
        oe = torch.cat((encode_context_left, encode_context_right), 2) 
        oe = self.output_layer(oe)
        oe = oe.mean(1)
        return oe

class SelfAttention(nn.Module):
    """docstring for SelfAttention"""
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(0)

        self.att_w = nn.Linear(input_dim, hidden_dim)
        self.att_v = nn.Parameter(torch.rand(hidden_dim))

        self.output_layer = nn.Sequential( 
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, input_dim)
                        )

    def forward(self, embed_input_left, embed_input_right):

        batch_size, num_context, seqlen, emb_dim = embed_input_left.size()

        # [batch, context, seq, dim]
        embed_input_left = self.dropout_layer(embed_input_left)
        embed_input_right = self.dropout_layer(embed_input_right)

        # [batch_size, context_num, seq_length, dim]
        left_right_context = torch.cat((embed_input_left, embed_input_right),2)
        #print(left_right_context.size())

        att_weight = torch.matmul(self.att_w(left_right_context), self.att_v)
        att_weight = nn.functional.softmax(att_weight, dim=2).view(batch_size, num_context, 2*seqlen, 1)
        #print(att_weight.size())
        
        oe = (left_right_context * att_weight).sum(2)

        oe = self.output_layer(oe)

        oe = oe.mean(1)

        return oe ,att_weight


class HierAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(HierAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(0)

        self.att_w = nn.Linear(input_dim, hidden_dim)
        self.att_v = nn.Parameter(torch.rand(hidden_dim))

        self.att_h = nn.Linear(input_dim, hidden_dim)
        self.att_hv = nn.Parameter(torch.rand(hidden_dim))

        self.output_layer = nn.Sequential( 
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, input_dim)
                        )

    def forward(self, embed_input_left, embed_input_right):

        batch_size, num_context, seqlen, emb_dim = embed_input_left.size()

        # [batch, context, seq, dim]
        embed_input_left = self.dropout_layer(embed_input_left)
        embed_input_right = self.dropout_layer(embed_input_right)

        # [batch_size, context_num, seq_length, dim]
        left_right_context = torch.cat((embed_input_left, embed_input_right),2)
        #print(left_right_context.size())


        att_weight = torch.matmul(self.att_w(left_right_context), self.att_v)
        att_weight = nn.functional.softmax(att_weight, dim=2).view(batch_size, num_context, 2*seqlen, 1)
        
        oe = (left_right_context * att_weight).sum(2)

        #print(oe.size())

        hier_att_weight = torch.matmul(self.att_h(oe), self.att_hv)
        #print(hier_att_weight.size())

        hier_att_weight =  nn.functional.softmax(hier_att_weight, dim=1).view(batch_size, num_context, 1)
        #print(hier_att_weight.size())

        oe = (oe * hier_att_weight).sum(1)

        oe = self.output_layer(oe)

        return oe, att_weight, hier_att_weight



class HierAttentionEnsemble(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(HierAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(0)

        self.att_w = nn.Linear(input_dim, hidden_dim)
        self.att_v = nn.Parameter(torch.rand(hidden_dim))

        self.att_h = nn.Linear(input_dim, hidden_dim)
        self.att_hv = nn.Parameter(torch.rand(hidden_dim))

        self.output_layer = nn.Sequential( 
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, input_dim)
                        )

    def forward(self, embed_input_left, embed_input_right):

        batch_size, num_context, seqlen, emb_dim = embed_input_left.size()

        # [batch, context, seq, dim]
        embed_input_left = self.dropout_layer(embed_input_left)
        embed_input_right = self.dropout_layer(embed_input_right)

        # [batch_size, context_num, seq_length, dim]
        left_right_context = torch.cat((embed_input_left, embed_input_right),2)
        #print(left_right_context.size())


        att_weight = torch.matmul(self.att_w(left_right_context), self.att_v)
        att_weight = nn.functional.softmax(att_weight, dim=2).view(batch_size, num_context, 2*seqlen, 1)
        
        oe = (left_right_context * att_weight).sum(2)

        #print(oe.size())

        hier_att_weight = torch.matmul(self.att_h(oe), self.att_hv)
        #print(hier_att_weight.size())

        hier_att_weight =  nn.functional.softmax(hier_att_weight, dim=1).view(batch_size, num_context, 1)
        #print(hier_att_weight.size())

        oe = (oe * hier_att_weight).sum(1)

        oe = self.output_layer(oe)

        return oe, att_weight, hier_att_weight
        

class ATTENTION(nn.Module):
    """docstring for ATTENTION"""
    def __init__(self, input_dim, hidden_dim):
        super(ATTENTION, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(0)
        self.left_context_encoder = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.right_context_encoder = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.att_w = nn.Linear(hidden_dim*2, hidden_dim)
        self.att_v = nn.Parameter(torch.rand(hidden_dim))
        self.output_layer = nn.Sequential( 
                            nn.Linear(hidden_dim*2, hidden_dim*2),
                            nn.ReLU(),
                            nn.Linear(hidden_dim*2, input_dim)
                        )

    def forward(self, embed_input_left, embed_input_right):
        # input: [batch, context, seq, emb]
        batch_size, num_context, seqlen, emb_dim = embed_input_left.size()
  
        # [batch, context, seq, dim] -> [batch*context, seq, dim]
        embed_input_left = embed_input_left.view(-1, seqlen, self.input_dim)
        embed_input_left = self.dropout_layer(embed_input_left)
        embed_input_right = embed_input_right.view(-1, seqlen, self.input_dim)
        embed_input_right = self.dropout_layer(embed_input_right)

        # hidden = (torch.zeros(1, batch_size*num_context, self.hidden_dim),
        #           torch.zeros(1, batch_size*num_context, self.hidden_dim))

        output_left, (final_hidden_state_left, final_cell_state_left) = self.left_context_encoder(embed_input_left) #, hidden)
        output_right,(final_hidden_state_right, final_cell_state_left) = self.right_context_encoder(embed_input_right) #, hidden)

        encode_context_left = final_hidden_state_left.view(-1, num_context, self.hidden_dim)
        encode_context_right = final_hidden_state_right.view(-1, num_context, self.hidden_dim)

        # concat + mean_pooling + fully_connect
        oe = torch.cat((encode_context_left, encode_context_right), 2)
        print(oe.size()) 
        att_weight = torch.matmul(self.att_w(oe), self.att_v)
        print(att_weight.size())
        att_weight = nn.functional.softmax(att_weight, dim=1).view(batch_size, num_context, 1)
        print(att_weight.size())
        oe = (oe * att_weight).sum(1)

        print("--------")

        oe = self.output_layer(oe)
        return oe

class BertEncoder(nn.Module):
    
    def __init__(self, bert_dir, model_type="base"):
        super(BertEncoder, self).__init__()
        self.model_type = model_type
        self.model = BertModel.from_pretrained(bert_dir)
        self.set_finetune("full")

    def set_finetune(self, finetune_type):
     
        if finetune_type == "none":
            for param in self.model.parameters():
                param.requires_grad = False
        elif finetune_type == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        elif finetune_type == "last":
            for param in self.model.parameters():
                param.require_grad = False
            for param in self.encoder.layer[-1].parameters():
                param.require_grad = True

    def forward(self, input_ids, mask=None):
        
        # [batch_size, context_num, seq_length]
        batch_size, context_num, seq_length = input_ids.size()
        flat_input_ids = input_ids.reshape(-1, input_ids.size(-1))
        flat_mask = mask.reshape(-1, mask.size(-1))
        pooled_cls = self.model(input_ids = flat_input_ids, attention_mask=flat_mask)[1]
        # [batch_size * context_num, dim]
        #print(pooled_cls.size())

        reshaped_pooled_cls = pooled_cls.view(batch_size, context_num, -1)
        # [batch_size, context_num, dim]
        output = reshaped_pooled_cls.mean(1)
        # [batch_size, dim]
        return output
        
    def get_output_dim(self):
        if self.model_type == "large":
            return 1024
        else:
            return 768

class Bert2Score(nn.Module):
    
    def __init__(self, encoder, bert_dir, hidden_dim, drop_prob):
        super(Bert2Score, self).__init__()
        self.hidden_dim = hidden_dim
        if "large" in encoder:
            self.encoder = BertEncoder(bert_dir, "large")
        else:
            self.encoder = BertEncoder(bert_dir)
 
        bert_dim = self.encoder.get_output_dim()
        self.mlp1 = nn.Linear(bert_dim, hidden_dim)
        self.mlp2 = nn.Linear(bert_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input_ids, masks):
        ## input: [batch_size, 2, context, seq]        
        left_ids = input_ids[:,0,:,:] 
        right_ids = input_ids[:,1,:,:]
        
        left_masks = masks[:,0,:,:]
        right_masks = masks[:,1,:,:]
        
        left_emb = self.encoder(left_ids, left_masks) 
        right_emb = self.encoder(right_ids, right_masks)
    
        # [batch_size, hidden_dim]   
        tran_left = self.mlp1(self.dropout(left_emb))
        tran_right = self.mlp2(self.dropout(right_emb))

        output = torch.einsum('ij,ij->i', [tran_left, tran_right])
        return output

class Context2Score(nn.Module):
    """docstring for Context2Score"""
    def __init__(self, encoder, input_dim, hidden_dim, device, multiple=False):
        super(Context2Score, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.device = device
        self.attention = False
        self.hier = False
        #self.name = encoder
        if 'lstm' in encoder:
            if multiple:
                self.encoder1 = nn.DataParallel(LSTM(input_dim, hidden_dim), device_ids=[0,1,2,3])
                self.encoder2 = nn.DataParallel(LSTM(input_dim, hidden_dim), device_ids=[0,1,2,3])
            else:
                self.encoder1 = LSTM(input_dim, hidden_dim).to(device)
                self.encoder2 = LSTM(input_dim, hidden_dim).to(device)
        elif 'attention' in encoder:
            if multiple:
                self.encoder1 = ATTENTION(input_dim, hidden_dim)
                self.encoder2 = ATTENTION(input_dim, hidden_dim)
            else:
                self.encoder1 = ATTENTION(input_dim, hidden_dim).to(device)
                self.encoder2 = ATTENTION(input_dim, hidden_dim).to(device)
        elif 'max' in encoder: 
            self.encoder1 = MEAN_Max(input_dim, hidden_dim).to(device)
            self.encoder2 = MEAN_Max(input_dim, hidden_dim).to(device)
        elif 'self' in encoder:
            #self.encoder1, self.atten1  = SelfAttention(input_dim, hidden_dim).to(device)
            self.encoder1  = SelfAttention(input_dim, hidden_dim).to(device)
            self.encoder2  = SelfAttention(input_dim, hidden_dim).to(device)
            self.attention = True

        elif 'han' in encoder:
            self.encoder1 = HierAttention(input_dim, hidden_dim).to(device)
            self.encoder2 = HierAttention(input_dim, hidden_dim).to(device)
            self.hier = True
                
        else:
            if multiple:
                self.encoder1 = MEAN(input_dim, hidden_dim)
                self.encoder2 = MEAN(input_dim, hidden_dim)
            else:
                self.encoder1 = MEAN(input_dim, hidden_dim).to(device)
                self.encoder2 = MEAN(input_dim, hidden_dim).to(device)

        
    def init_emb(self, w2v_weight):
        self.word_embedding = nn.Embedding.from_pretrained(w2v_weight, freeze=True)

    def forward(self, input_idx):
        # input: [batch, 2, context, 2, seq]
        
        embed_input1_left = self.word_embedding(input_idx[:, 0, :, 0]).to(self.device)
        embed_input1_right = self.word_embedding(input_idx[:, 0, :, 1]).to(self.device)
        embed_input2_left = self.word_embedding(input_idx[:, 1, :, 0]).to(self.device)
        embed_input2_right = self.word_embedding(input_idx[:, 1, :, 1]).to(self.device)

        if self.attention:
            embed_hypo, atten1 = self.encoder1(embed_input1_left, embed_input1_right)
            embed_hype, atten2  = self.encoder2(embed_input2_left, embed_input2_right)
            
            output = torch.einsum('ij,ij->i', [embed_hypo, embed_hype])
            return output, atten1, atten2

        elif self.hier:

            embed_hypo, atten1, hier_atten1 = self.encoder1(embed_input1_left, embed_input1_right)
            embed_hype, atten2, hier_atten2 = self.encoder2(embed_input2_left, embed_input2_right)

            output = torch.einsum('ij,ij->i', [embed_hypo, embed_hype])

            atten_w = (atten1, hier_atten1, atten2, hier_atten2)

            return output, atten_w 

        else:
            embed_hypo = self.encoder1(embed_input1_left, embed_input1_right)
            embed_hype = self.encoder2(embed_input2_left,embed_input2_right)
            output = torch.einsum('ij,ij->i', [embed_hypo, embed_hype])

            return output
      
