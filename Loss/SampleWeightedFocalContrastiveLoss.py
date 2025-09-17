# import torch
# import torch.nn as nn
# from torch.nn.functional import normalize
# '''
# Sample-Weighted Focal Contrastive (SWFC) Loss:  算的是一个bacth的
# 1. Divide training samples into positive and negative pairs to maximize 
# inter-class distances while minimizing intra-class distances;
# 2. Assign more importance to hard-to-classify positive pairs;
# 3. Assign more importance to minority classes. 
# '''
# class SampleWeightedFocalContrastiveLoss(nn.Module):

#     def __init__(self, temp_param, focus_param, sample_weight_param, dataset, class_counts, device):
#         '''
#         temp_param: control the strength of penalty on hard negative samples;
#         focus_param: forces the model to concentrate on hard-to-classify samples;
#         sample_weight_param: control the strength of penalty on minority classes;
#         dataset: MELD or IEMOCAP.
#         device: cpu or cuda. 
#         '''
#         super().__init__()
        
#         self.temp_param = temp_param   # t
#         self.focus_param = focus_param   # γ
#         self.sample_weight_param = sample_weight_param  # α
#         self.dataset = dataset
#         self.class_counts = class_counts
#         self.device = device

#         if self.dataset == 'MELD':
#             self.num_classes = 7
#         elif self.dataset == 'IEMOCAP':
#             self.num_classes = 6
#         else:
#             raise ValueError('Please choose either MELD or IEMOCAP')
        
#         self.class_weights = self.get_sample_weights()
    
#     #cxm
#     '''
#     Measure the correlation between two random variables X and Y using Soft-HGR maximum correlation. 
#     X_embs, Y_embs:current_features, feature_sets
    
#     '''
#     def soft_HGR_correlation(self, X_embs, Y_embs):
#         X_embs_mean = torch.mean(X_embs, dim = 0)
#         Y_embs_mean = torch.mean(Y_embs, dim = 0)
#         zero_mean_X_embs = X_embs - X_embs_mean
#         zero_mean_Y_embs = Y_embs - Y_embs_mean
#         X_Y_embs_expe = torch.sum(zero_mean_X_embs * zero_mean_Y_embs, dim = -1) / (zero_mean_X_embs.shape[0] - 1)
#         X_embs_corr = torch.cov(zero_mean_X_embs)
#         Y_embs_corr = torch.cov(zero_mean_Y_embs)
#         X_Y_embs_corr = torch.diagonal(X_embs_corr @ Y_embs_corr)
#         corr = X_Y_embs_expe - X_Y_embs_corr / 2
        
#         similarity_probs = torch.softmax(corr /  self.temp_param, dim = 0)
    
#         return similarity_probs

# #     '''
# #     Use dot-product to measure the similarity between feature pairs.
# #     '''
# #     def dot_product_similarity(self, current_features, feature_sets):
# #         similarity = torch.sum(current_features * feature_sets, dim = -1)
# #         similarity_probs = torch.softmax(similarity / self.temp_param, dim = 0)

# #         return similarity_probs

    
#     '''
#     Calculate the loss contributed from positive pairs.
#     '''
#     def positive_pairs_loss(self, similarity_probs):
#         pos_pairs_loss = torch.mean(torch.log(similarity_probs) * ((1 - similarity_probs)**self.focus_param), dim = 0)

#         return pos_pairs_loss


#     '''
#     Assign more importance to minority classes. 
#     '''
#     def get_sample_weights(self):
#         total_counts = torch.sum(self.class_counts, dim = -1)   #这里的 class_counts是算的全部样本的
#         class_weights = (total_counts / self.class_counts)**self.sample_weight_param 
#         class_weights = normalize(class_weights, dim = -1, p = 1.0)

#         return class_weights
       
#      #

#     def forward(self, features, labels):
#         self.num_samples = labels.shape[0]
#         self.feature_dim = features.shape[-1]

#         features = normalize(features, dim = -1)  # normalization helps smooth the learning process

#         batch_sample_weights = torch.FloatTensor([self.class_weights[label] for label in labels]).to(self.device)   #各个类别的权重

#         total_loss = 0.0
#         for i in range(self.num_samples):   #一个bacth里所有的样本，对应于公式里面的i,j循环    按照跑出来的结果，23x100--->2300,然后去掉标签为-1的，变成956
#             current_feature = features[i]
#             current_label = labels[i]
#             feature_sets = torch.cat((features[:i], features[i + 1:]), dim = 0)  #将features中索引为i的特征向量排除，然后将剩余的特征向量连接起来，形成一个新的特征集合。
#             label_sets = torch.cat((labels[:i], labels[i + 1:]), dim = 0)
#             expand_current_features = current_feature.expand(self.num_samples - 1, self.feature_dim).to(self.device)  #复制self.num_samples - 1次，扩展到与其他样本相同的维度。
#             similarity_probs = self.soft_HGR_correlation(expand_current_features, feature_sets)   #求出所有特征对之间的相似性
#             pos_similarity_probs = similarity_probs[label_sets == current_label]  # positive pairs with the same label找出同类别样本相似性
#             if len(pos_similarity_probs) > 0:
#                 pos_pairs_loss = self.positive_pairs_loss(pos_similarity_probs)
#                 weighted_pos_pairs_loss = pos_pairs_loss * batch_sample_weights[i]   #算的是bacth里面一个样本的
#                 total_loss += weighted_pos_pairs_loss
        
#         loss = - total_loss / self.num_samples    #除以的是总共是里面的N

#         return loss

import torch
import torch.nn as nn
from torch.nn.functional import normalize

'''
Sample-Weighted Focal Contrastive (SWFC) Loss:
1. Divide training samples into positive and negative pairs to maximize 
inter-class distances while minimizing intra-class distances;
2. Assign more importance to hard-to-classify positive pairs;
3. Assign more importance to minority classes. 
'''
class SampleWeightedFocalContrastiveLoss(nn.Module):

    def __init__(self, temp_param, focus_param, sample_weight_param, dataset, class_counts, device):
        '''
        temp_param: control the strength of penalty on hard negative samples;
        focus_param: forces the model to concentrate on hard-to-classify samples;
        sample_weight_param: control the strength of penalty on minority classes;
        dataset: MELD or IEMOCAP.
        device: cpu or cuda. 
        '''
        super().__init__()
        
        self.temp_param = temp_param
        self.focus_param = focus_param
        self.sample_weight_param = sample_weight_param
        self.dataset = dataset
        self.class_counts = class_counts
        self.device = device

        if self.dataset == 'MELD':
            self.num_classes = 7
        elif self.dataset == 'IEMOCAP':
            self.num_classes = 6
        else:
            raise ValueError('Please choose either MELD or IEMOCAP')
        
        self.class_weights = self.get_sample_weights()
    

    '''
    Use dot-product to measure the similarity between feature pairs.
    '''
    def dot_product_similarity(self, current_features, feature_sets):
        similarity = torch.sum(current_features * feature_sets, dim = -1)
        similarity_probs = torch.softmax(similarity / self.temp_param, dim = 0)

        return similarity_probs
    

    '''
    Calculate the loss contributed from positive pairs.
    '''
    def positive_pairs_loss(self, similarity_probs):
      
        pos_pairs_loss = torch.mean(torch.log(similarity_probs), dim = 0)
    
        return pos_pairs_loss


    '''
    Assign more importance to minority classes. 
    '''
    def get_sample_weights(self):
        total_counts = torch.sum(self.class_counts, dim = -1)
        class_weights = (total_counts / self.class_counts)**self.sample_weight_param
        class_weights = normalize(class_weights, dim = -1, p = 1.0)

        return class_weights
        

    def forward(self, features, labels):
        self.num_samples = labels.shape[0]
        self.feature_dim = features.shape[-1]

        features = normalize(features, dim = -1)  # normalization helps smooth the learning process

        batch_sample_weights = torch.FloatTensor([self.class_weights[label] for label in labels]).to(self.device)

        total_loss = 0.0
        for i in range(self.num_samples):
            current_feature = features[i]
            current_label = labels[i]
            feature_sets = torch.cat((features[:i], features[i + 1:]), dim = 0)
            label_sets = torch.cat((labels[:i], labels[i + 1:]), dim = 0)
            expand_current_features = current_feature.expand(self.num_samples - 1, self.feature_dim).to(self.device)
            similarity_probs = self.dot_product_similarity(expand_current_features, feature_sets)
            pos_similarity_probs = similarity_probs[label_sets == current_label]  # positive pairs with the same label
            if len(pos_similarity_probs) > 0:
                pos_pairs_loss = self.positive_pairs_loss(pos_similarity_probs)
                total_loss += pos_pairs_loss
        
        loss = - total_loss / self.num_samples

        return loss