import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import torch.nn as nn
torch.manual_seed(0)

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.criterion_bce = torch.nn.BCEWithLogitsLoss().to(self.args.device)  # For contrastive loss
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        ).to(self.args.device)
    def forward(self, x):
        features = self.model(x)
        logits = self.classifier(features)
        return features, logits

    def info_nce_loss(self, features):
        # 初始标签，一半0一半1
        labels = torch.cat([torch.zeros(self.args.batch_size), torch.ones(self.args.batch_size)], dim=0)
        #labels = torch.arange(2 * self.args.batch_size) % 2
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        #print(f"Initial labels: {labels}")  # 输出初始标签

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        assert similarity_matrix.shape == (self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size - 1)

        # select and combine multiple positives
        #print(similarity_matrix[labels.bool()])
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        #print(positives)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        # 输出标签和样本地址
        # print(f"Labels before logits: {labels}")
        # print(f"Positives: {positives}")
        # print(f"Negatives: {negatives}")

        logits = torch.cat([positives, negatives], dim=1)

        # update labels
        labels = torch.cat([torch.ones(positives.size(0), positives.size(1)), torch.zeros(positives.size(0), negatives.size(1))], dim=1).to(self.args.device)


        # print(f"logits shape: {logits.shape}")
        # print(f"labels shape: {labels.shape}")
        # print(f"logits content: {logits}")
        # print(f"labels content: {labels}")
        # print("Positives:", positives)
        # print("Negatives:", negatives)
        # print("Labels:", labels)

        logits = logits / self.args.temperature
        return logits, labels

    def load_checkpoint(checkpoint_path, model, classifier, optimizer):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        arch = checkpoint['arch']

        return model, classifier, optimizer, start_epoch, arch



    def train(self, train_loader):
        for epoch_counter in range(43):
            checkpoint_path = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            
            #checkpoint_path = 'checkpoint_30.pth.tar'  # 使用已保存的模型路径
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            
            # 如果检查点中包含分类器的状态字典
            if 'classifier_state_dict' in checkpoint:
                self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            
            total = 0
            correct = 0
            for image1, image2 in tqdm(train_loader):
                images = torch.cat([image1, image2], dim=0)
                images = images.to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    _, logits_cl = self.forward(images)
                    _, predict = torch.max(logits_cl, 1)
                    labels = torch.cat([torch.zeros(self.args.batch_size), torch.ones(self.args.batch_size)], dim=0).long()
                    labels = labels.to(self.args.device)
                    total += labels.size(0)
                    correct += (predict == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Model: {checkpoint_path}, Test Accuracy: {accuracy:.2f}%")

