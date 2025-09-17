import sys
sys.path.append('Loss')
sys.path.append('Model')
sys.path.append('Dataset')
from SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss
from SoftHGRLoss import SoftHGRLoss
from IEMOCAPDataset import IEMOCAPDataset
from MELDDataset import MELDDataset
from DRNN_MultiEMO_Model import MultiEMO
# from LSTM_MultiEMO_Model import MultiEMO
# from acme_Model import MultiEMO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from optparse import OptionParser
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data.sampler import SubsetRandomSampler
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import normalize
# from FocalLoss import FocalLoss
# from AutomaticWeightedLoss import AutomaticWeightedLoss
# from PolyCrossEntropy import Poly1CrossEntropyLoss


class TrainMultiEMO():

    def __init__(self, dataset, batch_size, num_epochs, learning_rate, weight_decay, 
                 num_layers, model_dim, num_heads, hidden_dim, dropout_rate, dropout_rec,
                 temp_param, focus_param, sample_weight_param, SWFC_loss_param, 
                 HGR_loss_param, CE_loss_param, multi_attn_flag, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dropout_rec = dropout_rec
        self.temp_param = temp_param
        self.focus_param = focus_param
        self.sample_weight_param = sample_weight_param
        self.SWFC_loss_param = SWFC_loss_param
        self.HGR_loss_param = HGR_loss_param
        self.CE_loss_param = CE_loss_param
        self.multi_attn_flag = multi_attn_flag
        self.device = device

        self.best_test_f1 = 0.0
        self.best_epoch = 1
        self.best_test_report = None

        self.get_dataloader()
        self.get_model()
        self.get_loss()
        self.get_optimizer()
     
        self.original_fc = nn.Linear(self.model_dim * 3, self.model_dim) #cxm
        self.text_fc = nn.Linear(768,self.model_dim)
        self.audio_fc = nn.Linear(512,self.model_dim)
        self.visual_fc = nn.Linear(1000,self.model_dim)

    def get_train_valid_sampler(self, train_dataset, valid = 0.1):   
        size = len(train_dataset)
        idx = list(range(size))
        split = int(valid * size)
        np.random.shuffle(idx)
        return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])   


    def get_dataloader(self, valid = 0.05):
        if self.dataset == 'IEMOCAP':
            train_dataset = IEMOCAPDataset(train = True)
            test_dataset = IEMOCAPDataset(train = False)
        elif self.dataset == 'MELD':
            train_dataset = MELDDataset(train = True)
            test_dataset = MELDDataset(train = False)

        train_sampler, valid_sampler = self.get_train_valid_sampler(train_dataset, valid)
        self.train_dataloader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, 
                                           sampler = train_sampler, collate_fn = train_dataset.collate_fn, num_workers = 0)
        self.valid_dataloader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, 
                                          sampler = valid_sampler,collate_fn = train_dataset.collate_fn, num_workers = 0)
        self.test_dataloader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, 
                                          collate_fn = test_dataset.collate_fn, shuffle = False, num_workers = 0)
    
    
    
    #计算整个数据集中各个类别的样本数量
    def get_class_counts(self):
        class_counts = torch.zeros(self.num_classes).to(self.device)

        for _, data in enumerate(self.train_dataloader):    #假设 batch_size=16,每个data里面包含16条样本
            _, _, _, _, _, padded_labels = [d.to(self.device) for d in data]
#             print("padded_labels.shape:",padded_labels.shape)  
            padded_labels = padded_labels.reshape(-1)   
            labels = padded_labels[padded_labels != -1]
            class_counts += torch.bincount(labels, minlength = self.num_classes) #统计真实标签中每个类别的样本数量
#             print("每个class_counts的shape",class_counts.shape,class_counts)
#         print("相加的class_counts的shape",class_counts.shape,class_counts)
        return class_counts
   
    #cxm
#     def get_sample_weights(self):
#         class_counts = self.get_class_counts()
#         total_counts = torch.sum(class_counts, dim = -1)
#         class_weights = (total_counts / class_counts)
#         class_weights = normalize(class_weights, dim = -1, p = 1.0)
#         return class_weights    
    
    def get_model(self):
        if self.dataset == 'IEMOCAP':
            self.num_classes = 6
            self.n_speakers = 2
        elif self.dataset == 'MELD':
            self.num_classes = 7
            self.n_speakers = 9

        roberta_dim = 768
        D_m_audio = 512
        D_m_visual = 1000
        listener_state = False
        D_e = self.model_dim 
        D_p = self.model_dim
        D_g = self.model_dim
        D_h = self.model_dim
        D_a = self.model_dim 
        context_attention = 'simple'
        hidden_dim = self.hidden_dim
        dropout_rate = self.dropout_rate 
        num_layers = self.num_layers
        num_heads = self.num_heads
       

        self.model = MultiEMO(self.dataset, self.multi_attn_flag, roberta_dim, hidden_dim, dropout_rate, num_layers, 
                                    self.model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, 
                                    D_h, self.num_classes, self.n_speakers, 
                                    listener_state, context_attention, D_a, self.dropout_rec, self.device).to(self.device)
        print(self.model)  
#         print("The model have {} paramerters in total".format(sum(x.numel() for x in self.model.parameters()))) 
#         total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         print("输出可训练参数的数量Total trainable parameters:", total_trainable_params)
        
    def get_loss(self):
        class_counts = self.get_class_counts()
        self.SWFC_loss = SampleWeightedFocalContrastiveLoss(self.temp_param, self.focus_param, 
                                                            self.sample_weight_param, self.dataset, class_counts, self.device)
        self.HGR_loss = SoftHGRLoss()
        self.CE_loss = nn.CrossEntropyLoss()


    def get_optimizer(self):
#         self.awl = AutomaticWeightedLoss(3) #cxm
#         print("The awl have {} paramerters in total".format(sum(x.numel() for x in self.awl.parameters()))) 
#         self.optimizer = optim.Adam([{'params': self.model.parameters()},{'params': self.awl.parameters()}], lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.95, patience = 5, threshold = 1e-6, verbose = True)

    '''
    训练神经网络通常是以多个 epochs 进行的，每个 epoch 表示将训练数据集完整地过一遍。而 batch_size 则表示每次迭代训练所使用的样本数。
    在一个 epoch 中会有多少个迭代（iterations），每个迭代处理一个大小为 batch_size 的样本集。因此，如果设定了一个特定的 batch_size，那么在每个 epoch 中将会有多少个迭代来完成整个训练数据集的训练。
    损失函数的计算通常是在每个迭代（或每个 batch）中进行的，而不是每个 epoch 结束后再计算一次。在每个迭代中，模型会根据当前的参数和输入数据计算出输出并与真实标签计算损失，然后利用反向传播算法更新模型参数。这个过程会持续多个迭代，直到完成整个 epoch。

    因此，损失函数是在每个迭代中根据当前的模型参数和数据计算得到的，通过反向传播算法来更新模型参数，而不是每个 epoch 结束后再计算一次。这样可以更快地收敛模型、更及时地调整参数，从而提高训练效率和模型性能。
    '''

    def train_or_eval_model_per_epoch(self, dataloader, train = True):  #每轮可能会进行多次迭代，每次迭代也就是每个bacth
        if train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        total_SWFC_loss, total_HGR_loss, total_CE_loss = 0.0, 0.0, 0.0
        all_labels, all_preds = [], []
        all_fc_outputs = []  #cxm
        all_original_fc_outputs = [ ] #cxm
      
        for _, data in enumerate(dataloader):   #在一个bacth里算loss！！！
            if train:
                self.optimizer.zero_grad() 

            padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels = [d.to(self.device) for d in data]      #每个data有batch_size个样本
            
#             print("model输入padded_texts.shape:",padded_texts.shape)  #cxm  
#             print("model输入padded_audios.shape:",padded_audios.shape)  #cxm
#             print("model输入padded_visuals.shape:",padded_visuals.shape)  #cxm
#             print("model输入padded_labels.shape:",padded_labels.shape)
            
            padded_labels = padded_labels.reshape(-1)
#             print("展开的padded_labels形状:",padded_labels.shape)
            labels = padded_labels[padded_labels != -1]
             
            #cxm
            self.original_fc = self.original_fc.to(device)
            self.text_fc = self.text_fc.to(device)
            self.audio_fc = self.audio_fc.to(device)
            self.visual_fc = self.visual_fc.to(device)

            padded_texts1 = self.text_fc(padded_texts)
            padded_audios1 = self.audio_fc(padded_audios)
            padded_visuals1 = self.visual_fc(padded_visuals)
            
            padded_texts1 =  padded_texts1.transpose(0, 1)
            padded_audios1 = padded_audios1.transpose(0, 1)
            padded_visuals1 = padded_visuals1.transpose(0, 1)
            
            padded_texts1 = padded_texts1.reshape(-1, padded_texts1.shape[-1])
            padded_texts1 = padded_texts1[padded_labels != -1]
            padded_audios1 = padded_audios1.reshape(-1, padded_audios1.shape[-1])
            padded_audios1 = padded_audios1[padded_labels != -1]
            padded_visuals1 = padded_visuals1.reshape(-1, padded_visuals1.shape[-1])
            padded_visuals1 = padded_visuals1[padded_labels != -1]
            original_fused_features = torch.cat((padded_texts1, padded_audios1, padded_visuals1), dim = -1)
            original_fc_outputs = self.original_fc(original_fused_features)

#             fused_text_features,fc_outputs, mlp_outputs = \
#                 self.model(padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels)
#             """full """
            fused_text_features, fused_audio_features, fused_visual_features, fc_outputs, mlp_outputs = \
                self.model(padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels)

#             print(" 模型输出fused_text_features.shape:", fused_text_features.shape)  #cxm
#             print(' 模型输出,SWFC_loss使用的fc_outputs.shape:',fc_outputs.shape)
#             print(" SWFC_loss使用的labels.shape:",labels.shape)
            
            soft_HGR_loss = self.HGR_loss(fused_text_features, fused_text_features, fused_text_features)
#             soft_HGR_loss = self.HGR_loss(fused_text_features, fused_audio_features, fused_visual_features)  
            SWFC_loss = self.SWFC_loss(fc_outputs, labels)
            CE_loss = self.CE_loss(mlp_outputs, labels)
           
#             loss = CE_loss * self.CE_loss_param  
            loss = SWFC_loss * self.SWFC_loss_param + CE_loss * self.CE_loss_param  
            
#             loss = self.awl(soft_HGR_loss, SWFC_loss, CE_loss)

    
            total_loss += loss.item()    
            total_HGR_loss += soft_HGR_loss.item()
            total_SWFC_loss += SWFC_loss.item()
            total_CE_loss += CE_loss.item()
            
    
            if train:
                loss.backward()     #一个bacth的损失
                self.optimizer.step()
                
            preds = torch.argmax(mlp_outputs, dim = -1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_fc_outputs.append(fc_outputs.detach().cpu().numpy()) #cxm
            all_original_fc_outputs.append(original_fc_outputs.detach().cpu().numpy()) #cxm

            
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_fc_outputs = np.concatenate(all_fc_outputs)  #cxm
        all_original_fc_outputs = np.concatenate(all_original_fc_outputs) #cxm
        
        avg_f1 = round(f1_score(all_labels, all_preds, average = 'weighted') * 100, 4)
        avg_acc = round(accuracy_score(all_labels, all_preds) * 100, 4)
        report = classification_report(all_labels, all_preds, digits = 4)

        return round(total_loss, 4), round(total_HGR_loss, 4), round(total_SWFC_loss, 4), round(total_CE_loss, 4), avg_f1, avg_acc, report, all_labels, all_preds, all_fc_outputs, all_original_fc_outputs  #cxm


    def train_or_eval_linear_model(self):
        train_losses, valid_losses = [], [] #cxm
        for e in range(self.num_epochs):     #跑好多轮，更新优化参数，每轮输出结果
            train_loss, train_HGR_loss, train_SWFC_loss, train_CE_loss, train_f1, train_acc, _ ,_ , _ ,_ ,_= self.train_or_eval_model_per_epoch(self.train_dataloader, train = True)
            train_losses.append(train_loss)  #cxm
            with torch.no_grad():
                valid_loss, valid_HGR_loss, valid_SWFC_loss, valid_CE_loss,valid_f1, valid_acc, _ ,_ , _ , _,_= self.train_or_eval_model_per_epoch(self.valid_dataloader, train = False)
                valid_losses.append(valid_loss) 
                test_loss, test_HGR_loss, test_SWFC_loss, test_CE_loss, test_f1, test_acc, test_report , all_labels, all_preds, all_fc_outputs, all_original_fc_outputs = self.train_or_eval_model_per_epoch(self.test_dataloader, train = False)
            print('Epoch {}, train loss: {}, train HGR loss: {}, train SWFC loss: {}, train CE loss: {}, train f1: {}, train acc: {}'.format(e + 1, train_loss, train_HGR_loss, train_SWFC_loss, train_CE_loss, train_f1, train_acc))
            print('Epoch {}, valid loss: {}, valid HGR loss: {}, valid SWFC loss: {}, valid CE loss: {}, valid f1: {}, valid acc: {}'.format(e + 1, valid_loss, valid_HGR_loss, valid_SWFC_loss, valid_CE_loss, valid_f1, valid_acc))
            print('Epoch {}, test loss: {}, test HGR loss: {}, test SWFC loss: {}, test CE loss: {}, test f1: {}, test acc: {}, '.format(e + 1, test_loss, test_HGR_loss, test_SWFC_loss, test_CE_loss, test_f1, test_acc))
            print(test_report)   
        
            self.scheduler.step(valid_loss)

            if test_f1 >= self.best_test_f1:
                self.best_test_f1 = test_f1
                self.best_epoch = e + 1
                self.best_test_report = test_report
                self.best_all_preds = all_preds  #cxm
                self.best_all_labels = all_labels      #cxm
                self.best_all_fc_outputs = all_fc_outputs  #cxm
  
        print('Best test f1: {} at epoch {}'.format(self.best_test_f1, self.best_epoch))
        print(self.best_test_report)
        
        #cxm
        print('self.best_all_fc_outputs.shape',self.best_all_fc_outputs.shape)
        print('self.best_all_labels.shape',self.best_all_labels.shape)
        print('all_original_fc_outputs.shape',all_original_fc_outputs.shape)
        print('all_original_fc_outputs.shape',all_original_fc_outputs.shape)
        
        np.save('best_all_fc_outputs.npy', self.best_all_fc_outputs)
        np.save('best_all_labels.npy', self.best_all_labels)
        np.save('all_original_fc_outputs.npy', all_original_fc_outputs)
        
        #cxm
        if self.dataset == 'IEMOCAP':

            label_mapping = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'anger', 4: 'excited', 5: 'frustrated'}
            color_mapping = {'happy': 'red', 'sad': 'mediumpurple', 'neutral': 'orange', 'anger': 'gold', 'excited': 'darkgreen', 'frustrated': 'yellowgreen'}
            tsne = TSNE(n_components=2, random_state=1)
            embedded_data = tsne.fit_transform(all_original_fc_outputs)
            plt.figure(figsize=(8, 6))
            for label in set(self.best_all_labels):
                indices = [i for i, l in enumerate(self.best_all_labels) if l == label]
                plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label_mapping[label], color=color_mapping[label_mapping[label]])
            plt.legend(title='Class', loc='lower right')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            file_name_7 ='original_TSNE_iemocap.png'
            plt.savefig(file_name_7)
            plt.show()
            
            
            label_mapping = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'anger', 4: 'excited', 5: 'frustrated'}
            color_mapping = {'happy': 'red', 'sad': 'mediumpurple', 'neutral': 'orange', 'anger': 'gold', 'excited': 'darkgreen', 'frustrated': 'yellowgreen'}
            tsne = TSNE(n_components=2, random_state=1)
            embedded_data = tsne.fit_transform(self.best_all_fc_outputs)
            plt.figure(figsize=(8, 6))
            for label in set(self.best_all_labels):
                indices = [i for i, l in enumerate(self.best_all_labels) if l == label]
                plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label_mapping[label], color=color_mapping[label_mapping[label]])
            plt.legend(title='Class', loc='lower right')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            file_name ='TSNE_iemocap.png'
            plt.savefig(file_name)
            plt.show()
            
     
            cm = confusion_matrix(self.best_all_labels, self.best_all_preds)
            print('Confusion Matrix:\n',cm)
            np.savetxt('confusion_matrix_i.txt', cm, fmt='%d')
            label_mapping = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'anger', 4: 'excited', 5: 'frustrated'}
            labels = [label_mapping[i] for i in range(len(label_mapping))]
            print(labels)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.yticks(rotation=0)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            file_name_1 ='confusion_matrix_iemocap.png'
            plt.savefig(file_name_1)
            plt.show()
 
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, self.num_epochs + 1), train_losses, label='Training Loss')
            plt.plot(range(1, self.num_epochs + 1), valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            file_name_2 ='TV_Loss_iemocap.png'
            plt.savefig(file_name_2)
            plt.show()
  
            with open('train_losses_i.txt', 'w') as f:
                for loss in train_losses:
                    f.write("%f\n" % loss)
            with open('valid_losses_i.txt', 'w') as f:
                for loss in valid_losses:
                    f.write("%f\n" % loss)
            
        elif self.dataset == 'MELD':
            
            label_mapping = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'disgust', 6:'anger'}
            color_mapping = {'neutral': 'orange', 'surprise': 'salmon', 'fear': 'yellow', 'sadness': 'chartreuse', 'joy': 'green', 'disgust': 'cornflowerblue', 'anger': 'red'}
            tsne = TSNE(n_components=2, random_state=1)
            embedded_data = tsne.fit_transform(all_original_fc_outputs)
            plt.figure(figsize=(8, 6))
            for label in set(self.best_all_labels):
                indices = [i for i, l in enumerate(self.best_all_labels) if l == label]
                plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label_mapping[label], color=color_mapping[label_mapping[label]])
            plt.legend(title='Class', loc='lower right')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            file_name_7 ='original_TSNE_meld.png'
            plt.savefig(file_name_7)
            plt.show()  
            
            
            label_mapping = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'disgust', 6:'anger'}
            color_mapping = {'neutral': 'orange', 'surprise': 'salmon', 'fear': 'yellow', 'sadness': 'chartreuse', 'joy': 'green', 'disgust': 'cornflowerblue', 'anger': 'red'}

            tsne = TSNE(n_components=2, random_state=1)
            embedded_data = tsne.fit_transform(self.best_all_fc_outputs)

            plt.figure(figsize=(8, 6))
            for label in set(self.best_all_labels):
                indices = [i for i, l in enumerate(self.best_all_labels) if l == label]
                plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label_mapping[label], color=color_mapping[label_mapping[label]])
            plt.legend(title='Class', loc='lower right')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            file_name_3 ='TSNE_meld.png'
            plt.savefig(file_name_3)
            plt.show()
          
        
            cm = confusion_matrix(self.best_all_labels, self.best_all_preds)
            print('Confusion Matrix:\n',cm)
            np.savetxt('confusion_matrix_m.txt', cm, fmt='%d')
            label_mapping = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'disgust', 6:'anger'}
            labels = [label_mapping[i] for i in range(len(label_mapping))]
            print(labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            print('!!!!！')
            file_name_4 ='confusion_matrix_meld.png'
            plt.savefig(file_name_4)
            plt.show()
            
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, self.num_epochs + 1), train_losses, label='Training Loss')
            plt.plot(range(1, self.num_epochs + 1), valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            file_name_5 ='TV_Loss_meld.png'
            plt.savefig(file_name_5)
            plt.show()
            
            with open('train_losses_m.txt', 'w') as f:
                for loss in train_losses:
                    f.write("%f\n" % loss)
            with open('valid_losses_m.txt', 'w') as f:
                for loss in valid_losses:
                    f.write("%f\n" % loss)
            
            
def get_args():
    parser = OptionParser()
    parser.add_option('--dataset', dest = 'dataset', default = 'MELD', type = 'str', help = 'MELD or IEMOCAP')
    parser.add_option('--batch_size', dest = 'batch_size', default = 64, type = 'int', help = '64 for IEMOCAP and 100 for MELD')
    parser.add_option('--num_epochs', dest = 'num_epochs', default = 100, type = 'int', help = 'number of epochs')
    parser.add_option('--learning_rate', dest = 'learning_rate', default = 0.0001, type = 'float', help = 'learning rate')
    parser.add_option('--weight_decay', dest = 'weight_decay', default = 0.00001, type = 'float', help = 'weight decay parameter')
    parser.add_option('--num_layers', dest = 'num_layers', default = 6, type = 'int', help = 'number of layers in MultiAttn')
    parser.add_option('--model_dim', dest = 'model_dim', default = 256, type = 'int', help = 'model dimension in MultiAttn')
    parser.add_option('--num_heads', dest = 'num_heads', default = 4, type = 'int', help = 'number of heads in MultiAttn')
    parser.add_option('--hidden_dim', dest = 'hidden_dim', default = 1024, type = 'int', help = 'hidden dimension in MultiAttn')
    parser.add_option('--dropout_rate', dest = 'dropout_rate', default = 0, type = 'float', help = 'dropout rate')
    parser.add_option('--dropout_rec', dest = 'dropout_rec', default = 0, type = 'float', help = 'dropout rec')
    parser.add_option('--temp_param', dest = 'temp_param', default = 0.8, type = 'float', help = 'temperature parameter of SWFC loss')
    parser.add_option('--focus_param', dest = 'focus_param', default = 2.0, type = 'float', help = 'focusing parameter of SWFC loss')
    parser.add_option('--sample_weight_param', dest = 'sample_weight_param', default = 0.8, type = 'float', help = 'sample-weight parameter of SWFC loss')
    parser.add_option('--SWFC_loss_param', dest = 'SWFC_loss_param', default = 0.4, type = 'float', help = 'coefficient of SWFC loss')
    parser.add_option('--HGR_loss_param', dest = 'HGR_loss_param', default = 0.3, type = 'float', help = 'coefficient of Soft-HGR loss')
    parser.add_option('--CE_loss_param', dest = 'CE_loss_param', default = 0.3, type = 'float', help = 'coefficient of Cross Entropy loss')
    parser.add_option('--multi_attn_flag', dest = 'multi_attn_flag', default = True, help = 'Multimodal fusion')

    (options, _) = parser.parse_args()

    return options




def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




if __name__ == '__main__':
    torch.cuda.empty_cache()  #cxm
    args = get_args()
    dataset = args.dataset
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    model_dim = args.model_dim
    num_heads = args.num_heads
    hidden_dim = args.hidden_dim
    dropout_rate = args.dropout_rate
    dropout_rec = args.dropout_rec
    temp_param = args.temp_param
    focus_param = args.focus_param
    sample_weight_param = args.sample_weight_param
    SWFC_loss_param = args.SWFC_loss_param
    HGR_loss_param = args.HGR_loss_param
    CE_loss_param = args.CE_loss_param
    multi_attn_flag = args.multi_attn_flag
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args) #cxm
 
    seed = 12345
    set_seed(seed)

    multiemo_train = TrainMultiEMO(dataset, batch_size, num_epochs, learning_rate, 
                                   weight_decay, num_layers, model_dim, num_heads, hidden_dim, 
                                   dropout_rate, dropout_rec,temp_param, focus_param, sample_weight_param, 
                                   SWFC_loss_param, HGR_loss_param, CE_loss_param, multi_attn_flag, device)
    multiemo_train.train_or_eval_linear_model()

        