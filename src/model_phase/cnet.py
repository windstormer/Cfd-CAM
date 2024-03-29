import torch
import torch.optim as optim
import os
from datetime import datetime
from collections import OrderedDict

from models import *
from loss import *
from tqdm import tqdm
from utils import *
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

class CNet(object):
    def __init__(self, args, train_loader, val_loader, log_path, record_path, model_name, gpuid):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_path = log_path
        
        if args.model_type == "Res18":
            model = Res18_Classifier()
        elif args.model_type == "Res50":
            model = Res50_Classifier()

        if args.encoder_pretrained_path != None:
            encoder_pretrained_path = os.path.join(args.project_path, "record", args.encoder_pretrained_path, "model", "encoder.pth")
            model.load_encoder_pretrain_weight(encoder_pretrained_path)

        if args.pretrained_path != None:
            pretrained_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "model.pth")
            model.load_pretrain_weight(pretrained_path)
        # model = SSLModel(None)

        self.lr = args.learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        # self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0.000005)
        
        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
        # pretrain_model_path = os.path.join(project_path, record_path, "weights", "embedder.pth")
        self.model = model.to('cuda')

        self.loss = nn.BCEWithLogitsLoss()
        self.model_name = model_name
        self.project_path = args.project_path
        self.record_path = record_path

    def run(self):
        log_file = open(self.log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        train_record = {'auc':[], 'loss':[]}
        val_record = {'auc':[], 'loss':[]}
        best_score = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, sensitivity, specificity, train_auc = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['auc'].append(train_auc)

            self.scheduler.step()
            
            log_file = open(self.log_path, "a")
            log_file.writelines(
            f'Epoch {epoch:4d}/{self.epochs:4d} | Cur lr: {self.scheduler.get_last_lr()[0]} | Train Loss: {train_loss}, Train Acc: {train_acc}, AUC: {train_auc}\n')

            val_loss, val_acc, sensitivity, specificity, val_auc = self.val(epoch)
            val_record['loss'].append(val_loss)
            val_record['auc'].append(val_auc)
            log_file = open(self.log_path, "a")
            log_file.writelines(
            f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss}, Val Acc: {val_acc}, AUC: {val_auc}, Sensitivity: {sensitivity}, Specificity: {specificity}\n")

            cur_score = val_auc
            if cur_score > best_score:
                best_score = cur_score
                log_file.writelines(f"Save model at Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss}, Val Acc: {val_acc}, AUC: {val_auc}\n")
                model_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "model.pth")
                torch.save(self.model.state_dict(), model_path)
            log_file.close()

            # parameter_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "model.pth")
            # torch.save(self.model.state_dict(), parameter_path)
            log_file = open(self.log_path, "a")
            log_file.writelines(str(datetime.now())+"\n")
            log_file.close()
        save_chart(self.epochs, train_record['auc'], val_record['auc'], os.path.join(self.project_path, self.record_path, self.model_name, "auc.png"), name='auc')
        save_chart(self.epochs, train_record['loss'], val_record['loss'], os.path.join(self.project_path, self.record_path, self.model_name, "loss.png"), name='loss')

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        train_labels = []
        pred_results = []
        # out_results = []

        for case_batch, label_batch in train_bar:
            self.optimizer.zero_grad()
            loss, pred_batch = self.step(case_batch, label_batch)
            loss.backward()
            self.optimizer.step()
            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
            pred_results.append(pred_batch)
            train_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(train_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(train_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def step(self, data_batch, label_batch):
        _, _, logit = self.model(data_batch.cuda())
        loss = self.loss(logit, label_batch.cuda())
        pred = torch.sigmoid(logit)
        pred = pred.detach().cpu()
        return loss, pred

            
    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        val_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for case_batch, label_batch in val_bar:
                loss, pred_batch = self.step(case_batch, label_batch)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
                pred_results.append(pred_batch)
                val_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(val_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(val_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def test(self, loader, load_model=None):
        self.model.eval()
        test_bar = tqdm(loader)
        total_loss, total_num = 0.0, 0
        test_labels = []
        pred_results = []
        with torch.no_grad():
            for case_batch, label_batch in test_bar:
                loss, pred_batch = self.step(case_batch, label_batch)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                pred_results.append(pred_batch)
                test_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(test_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(test_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def evaluate(self, labels, pred):
        # fpr, tpr, threshold = roc_curve(labels, pred, pos_label=1)
        # fpr_optimal, tpr_optimal, threshold_optimal = self.optimal_thresh(fpr, tpr, threshold)
        out_results = [pred > 0.5 for pred in pred]
        auc_score = roc_auc_score(labels, pred)

        tn, fp, fn, tp = confusion_matrix(labels, out_results, labels=[0,1]).ravel()
        acc = (tp+tn) / (tn+fp+fn+tp)
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        return acc, sensitivity, specificity, auc_score

    # def optimal_thresh(self, fpr, tpr, thresholds, p=0):
    #     loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    #     idx = np.argmin(loss, axis=0)
    #     return fpr[idx], tpr[idx], thresholds[idx]