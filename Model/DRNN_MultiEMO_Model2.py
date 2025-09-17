from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb
        return x
    
"""
full setting
"""
class MultiEMO(nn.Module):

    def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
                 model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
        super().__init__()

        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag
        
        self.pos_emb = PositionalEncoding(model_dim)

        self.text_fc = nn.Linear(roberta_dim, model_dim)
        self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)

        self.audio_fc = nn.Linear(D_m_audio, model_dim)
        self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        
        self.visual_fc = nn.Linear(D_m_visual, model_dim)
        self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        
        self.multiattn = MultiAttnModel(num_layers, model_dim, num_heads, hidden_dim, dropout)
        
        self.linear_cat = nn.Linear(2 * model_dim, model_dim)

        self.fc = nn.Linear(model_dim * 3, model_dim)
        self.dropout = nn.Dropout(0.2)


        if self.dataset == 'MELD':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

 
    def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):
    
        text_features = self.text_fc(texts)
        text_features = self.pos_emb(text_features.permute(1, 0, 2)).transpose(1,0)  #[seq_len,bacth,,dim]
        if self.dataset == 'IEMOCAP':
            text_features = self.text_dialoguernn(text_features, speaker_masks, utterance_masks)

        audio_features = self.audio_fc(audios)
        audio_features = self.pos_emb(audio_features.permute(1, 0, 2)).transpose(1,0)
        audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)

        visual_features = self.visual_fc(visuals)
        visual_features = self.pos_emb(visual_features.permute(1, 0, 2)).transpose(1,0)
        visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)

        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        if self.multi_attn_flag == True:
            ta_features, tv_features, at_features, vt_features = self.multiattn(text_features, audio_features, visual_features) #输入形状(batct_size,seq_len,dim)
            
#             print("注意力之后的ta_features形状:",ta_features.shape)

            ta_features = ta_features.reshape(-1, ta_features.shape[-1])
            ta_features = ta_features[padded_labels != -1]

            tv_features = tv_features.reshape(-1, tv_features.shape[-1])
            tv_features = tv_features[padded_labels != -1]

            fused_text_features = torch.cat((ta_features, tv_features), dim = -1)
            fused_text_features = self.linear_cat(fused_text_features)
           

            at_features = at_features.reshape(-1, at_features.shape[-1])
            at_features = at_features[padded_labels != -1]

            fused_audio_features = at_features

            vt_features = vt_features.reshape(-1,vt_features.shape[-1])
            vt_features = vt_features[padded_labels != -1]

            fused_visual_features = vt_features
            
        else:   #去掉fusion
            fused_text_features, fused_audio_features, fused_visual_features = text_features, audio_features, visual_features
            
            fused_text_features = fused_text_features.reshape(-1, fused_text_features.shape[-1])
            fused_text_features = fused_text_features[padded_labels != -1]
            fused_audio_features = fused_audio_features.reshape(-1, fused_audio_features.shape[-1])
            fused_audio_features = fused_audio_features[padded_labels != -1]
            fused_visual_features = fused_visual_features.reshape(-1, fused_visual_features.shape[-1])
            fused_visual_features = fused_visual_features[padded_labels != -1]

        
        fused_features = torch.cat((fused_text_features, fused_audio_features, fused_visual_features), dim = -1)
#         fused_features = fused_text_features + fused_audio_features + fused_visual_features

        fc_outputs = self.fc(fused_features)
        mlp_outputs = self.mlp(fc_outputs)

        return fused_text_features, fused_audio_features, fused_visual_features, fc_outputs, mlp_outputs

# """
# text setting
# """
# class MultiEMO(nn.Module):

#     def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
#                  model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
#         super().__init__()

#         self.dataset = dataset
#         self.multi_attn_flag = multi_attn_flag

#         self.text_fc = nn.Linear(roberta_dim, model_dim)
#         self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)

#         self.fc = nn.Linear(model_dim, model_dim)

#         if self.dataset == 'MELD':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
#         elif self.dataset == 'IEMOCAP':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

 
#     def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):
#         text_features = self.text_fc(texts)
#         # We empirically find that additional context modeling leads to improved model performances on IEMOCAP   
# #         if self.dataset == 'IEMOCAP':
#         text_features = self.text_dialoguernn(text_features, speaker_masks, utterance_masks)

#         text_features = text_features.transpose(0, 1)
        

#         fused_text_features = text_features
            
#         fused_text_features = fused_text_features.reshape(-1, fused_text_features.shape[-1])
#         fused_text_features = fused_text_features[padded_labels != -1]
           
#         fused_features = fused_text_features
        
#         fc_outputs = self.fc(fused_features)
#         mlp_outputs = self.mlp(fc_outputs)

#         return fused_text_features, fc_outputs, mlp_outputs


# """
# audio setting
# """
# class MultiEMO(nn.Module):

#     def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
#                  model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
#         super().__init__()

#         self.dataset = dataset
#         self.multi_attn_flag = multi_attn_flag

#         self.audio_fc = nn.Linear(D_m_audio, model_dim)
#         self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)

#         self.fc = nn.Linear(model_dim , model_dim)

#         if self.dataset == 'MELD':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
#         elif self.dataset == 'IEMOCAP':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

 
#     def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):

#         audio_features = self.audio_fc(audios)
#         audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)

#         audio_features = audio_features.transpose(0, 1)

#         fused_audio_features = audio_features
#         fused_audio_features = fused_audio_features.reshape(-1, fused_audio_features.shape[-1])
#         fused_audio_features = fused_audio_features[padded_labels != -1]
        
#         fused_features =  fused_audio_features
       
#         fc_outputs = self.fc(fused_features)
#         mlp_outputs = self.mlp(fc_outputs)

#         return fused_audio_features, fc_outputs, mlp_outputs


# """
# visual setting
# """

# class MultiEMO(nn.Module):

#     def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
#                  model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
#         super().__init__()

#         self.dataset = dataset
#         self.multi_attn_flag = multi_attn_flag
        
#         self.visual_fc = nn.Linear(D_m_visual, model_dim)
        
#         self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)
       
#         self.fc = nn.Linear(model_dim , model_dim)

#         if self.dataset == 'MELD':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
#         elif self.dataset == 'IEMOCAP':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

 
#     def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):

#         visual_features = self.visual_fc(visuals)
#         visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)

#         visual_features = visual_features.transpose(0, 1)

#         fused_visual_features = visual_features
            
#         fused_visual_features = fused_visual_features.reshape(-1, fused_visual_features.shape[-1])
#         fused_visual_features = fused_visual_features[padded_labels != -1]
        
#         fused_features =  fused_visual_features
        
#         fc_outputs = self.fc(fused_features)
#         mlp_outputs = self.mlp(fc_outputs)

#         return  fused_visual_features, fc_outputs, mlp_outputs


# '''
# t+v
# '''

# class MultiEMO(nn.Module):

#     def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
#                  model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
#         super().__init__()

#         self.dataset = dataset
#         self.multi_attn_flag = multi_attn_flag

#         self.text_fc = nn.Linear(roberta_dim, model_dim)
#         self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)
        
#         self.visual_fc = nn.Linear(D_m_visual, model_dim)
#         self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)
        
#         self.fc = nn.Linear(model_dim * 2, model_dim)

#         if self.dataset == 'MELD':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
#         elif self.dataset == 'IEMOCAP':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

 
#     def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):
        
#         text_features = self.text_fc(texts)
#         text_features = self.text_dialoguernn(text_features, speaker_masks, utterance_masks)

#         visual_features = self.visual_fc(visuals)
#         visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)

#         text_features = text_features.transpose(0, 1)
#         visual_features = visual_features.transpose(0, 1)

#         fused_text_features, fused_visual_features = text_features,visual_features
            
#         fused_text_features = fused_text_features.reshape(-1, fused_text_features.shape[-1])
#         fused_text_features = fused_text_features[padded_labels != -1]
#         fused_visual_features = fused_visual_features.reshape(-1, fused_visual_features.shape[-1])
#         fused_visual_features = fused_visual_features[padded_labels != -1]

#         fused_features = torch.cat((fused_text_features, fused_visual_features), dim = -1)
        
#         fc_outputs = self.fc(fused_features)
#         mlp_outputs = self.mlp(fc_outputs)

#         return fused_text_features, fc_outputs, mlp_outputs

# """
# t+a
# """
# class MultiEMO(nn.Module):

#     def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
#                  model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
#         super().__init__()

#         self.dataset = dataset
#         self.multi_attn_flag = multi_attn_flag

#         self.text_fc = nn.Linear(roberta_dim, model_dim)
#         self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)

#         self.audio_fc = nn.Linear(D_m_audio, model_dim)
#         self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)

#         self.fc = nn.Linear(model_dim * 2, model_dim)

#         if self.dataset == 'MELD':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
#         elif self.dataset == 'IEMOCAP':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

 
#     def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):
#         text_features = self.text_fc(texts) 
#         text_features = self.text_dialoguernn(text_features, speaker_masks, utterance_masks)

#         audio_features = self.audio_fc(audios)
#         audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)

#         text_features = text_features.transpose(0, 1)
#         audio_features = audio_features.transpose(0, 1)


#         fused_text_features, fused_audio_features = text_features, audio_features
            
#         fused_text_features = fused_text_features.reshape(-1, fused_text_features.shape[-1])
#         fused_text_features = fused_text_features[padded_labels != -1]
#         fused_audio_features = fused_audio_features.reshape(-1, fused_audio_features.shape[-1])
#         fused_audio_features = fused_audio_features[padded_labels != -1]
           
#         fused_features = torch.cat((fused_text_features, fused_audio_features), dim = -1)
        
#         fc_outputs = self.fc(fused_features)
#         mlp_outputs = self.mlp(fc_outputs)

#         return fused_text_features, fc_outputs, mlp_outputs
    
# """
# a+v
# """
# class MultiEMO(nn.Module):

#     def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
#                  model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
#         super().__init__()

#         self.dataset = dataset
#         self.multi_attn_flag = multi_attn_flag

#         self.audio_fc = nn.Linear(D_m_audio, model_dim)
#         self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)
        
#         self.visual_fc = nn.Linear(D_m_visual, model_dim)
#         self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
#                  n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
#                  dropout, device)
        
#         self.fc = nn.Linear(model_dim * 2, model_dim)

#         if self.dataset == 'MELD':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
#         elif self.dataset == 'IEMOCAP':
#             self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
 
#     def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):
       
#         audio_features = self.audio_fc(audios)
#         audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)

#         visual_features = self.visual_fc(visuals)
#         visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)

#         audio_features = audio_features.transpose(0, 1)
#         visual_features = visual_features.transpose(0, 1)
        
#         fused_audio_features, fused_visual_features =  audio_features, visual_features
            
#         fused_audio_features = fused_audio_features.reshape(-1, fused_audio_features.shape[-1])
#         fused_audio_features = fused_audio_features[padded_labels != -1]
#         fused_visual_features = fused_visual_features.reshape(-1, fused_visual_features.shape[-1])
#         fused_visual_features = fused_visual_features[padded_labels != -1]

        
#         fused_features = torch.cat((fused_audio_features, fused_visual_features), dim = -1)
        
#         fc_outputs = self.fc(fused_features)
#         mlp_outputs = self.mlp(fc_outputs)

#         return fused_audio_features,fc_outputs, mlp_outputs





