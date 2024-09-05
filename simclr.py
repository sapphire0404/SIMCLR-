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

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # Save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        lambda_l2 = 0.001

        for epoch_counter in range(self.args.epochs):
            epoch_loss = 0  # Record the total loss for each epoch
            for image1, image2 in tqdm(train_loader):
                images = torch.cat([image1, image2], dim=0)  # Concatenate two images
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features, logits_cl = self.forward(images)
                    logits_co, labels = self.info_nce_loss(features)
                    contrastive_loss = self.criterion_bce(logits_co, labels)
                    labels = torch.cat([torch.zeros(self.args.batch_size), torch.ones(self.args.batch_size)], dim=0).long()
                    labels = labels.to(self.args.device)
                    classification_loss = self.criterion(logits_cl, labels)
                    loss = contrastive_loss + classification_loss
                    #print(contrastive_loss)
                    #print(classification_loss)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                epoch_loss += loss.item()  # Accumulate loss for each iteration

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)

                n_iter += 1

            # Print the average loss for each epoch
            print(f"Epoch [{epoch_counter + 1}/{self.args.epochs}], Loss: {epoch_loss / len(train_loader)}")

            # Warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            logging.info("Training has finished one epoch.")

            # Save the model at the end of each epoch
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter + 1)
            save_checkpoint({
                'epoch': epoch_counter + 1,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=checkpoint_name)

            print(f"Model saved as {checkpoint_name}")
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


