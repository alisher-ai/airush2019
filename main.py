import copy
from torchvision import transforms as tr
import os
import torch
import torch.nn as nn
from numpy import random as rand
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import train_dataloader
from dataloader import AIRushDataset
from fastai import *
from fastai.vision import *


class MixUpLoss(Module):
    "Adapt the loss function `crit` to go with mixup."

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else:
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = (loss1 * target[:, 2] + loss2 * (1 - target[:, 2])).mean()
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

def to_np(t):
    return t.cpu().detach().numpy()


def bind_model(model_nsml):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
            'model': model_nsml.state_dict(),
        }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])

    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)

        input_size = 224  # you can change this according to your model.
        batch_size = 200  # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0

        dataloader = DataLoader(
            AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                          transform=tr.Compose(
                              [
                                  tr.Resize((input_size, input_size)),
                                  tr.ToTensor(),
                                  tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ]
                          ),
                          labels=None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        for batch_idx, image in enumerate(dataloader):
            image = image.to(device)
            output = model_nsml(image).double()

            output_prob = F.softmax(output, dim=1)
            predict = np.argmax(to_np(output_prob), axis=1)
            predict_list.append(predict)
            print(predict_list)

        predict_vector = np.concatenate(predict_list, axis=0)
        return predict_vector  # this return type should be a numpy array which has shape of (138343, 1)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)

    # custom args
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--resnet', default=False)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=350)  # Fixed
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    epsilon = 4./350
    last_value = 1
    heuristic_vector = []
    for i in range(args.output_size):
        last_value += epsilon
        heuristic_vector.append(last_value)
    heuristic_vector = heuristic_vector[::-1]

    torch.manual_seed(args.seed)
    device = args.device

    if args.resnet:
        assert args.input_size == 224
        model = Resnet(args.output_size)
    else:
        model = Baseline(args.hidden_size, args.output_size)

    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), args.learning_rate)
    # optimizer = optim.SparseAdam(model.parameters(), args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    # criterion = nn.CrossEntropyLoss()  # multi-class classification task
    model = model.to(device)
    model.train()

    # DONOTCHANGE: They are reserved for nsml
    bind_model(model)
    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        # Warning: Do not load data before this line
        train_dataloader, val_dataloader = train_dataloader(args.input_size, args.batch_size, args.num_workers)
        for epoch_idx in range(1, args.epochs + 1):
            scheduler.step()
            print("==================================================")
            print("Epoch: {} \t lr: {}".format(epoch_idx, scheduler.get_lr()))
            model.train()
            total_loss = 0
            total_correct = 0
            criterion = LabelSmoothingCrossEntropy()

            for batch_idx, (image, tags) in enumerate(train_dataloader):
                if rand.rand() > 0.5:
                    alpha = 0.4
                    lambd = np.random.beta(alpha, alpha, tags.size(0))
                    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
                    lambd = image.new(lambd)
                    shuffle = torch.randperm(tags.size(0)).to(tags.device)
                    x1, y1 = image[shuffle], tags[shuffle]

                    out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
                    new_input = (last_input * lambd.view(out_shape) + x1 * (1 - lambd).view(out_shape))
                    new_target = torch.cat([last_target[:, None].float(), y1[:, None].float(), lambd[:, None].float()],1)
                    image = new_input
                    tags = new_target
                    criterion = MixUpLoss(criterion)

                optimizer.zero_grad()
                image = image.to(device)
                tags = tags.to(device)
                output = model(image).double()
                loss = criterion(output, tags)
                loss.backward()
                optimizer.step()

                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_np(output_prob), axis=1)
                label_vector = to_np(tags)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)

                if batch_idx % args.log_interval == 0:
                    print('Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(batch_idx,
                                                                                         len(train_dataloader),
                                                                                         loss.item(),
                                                                                         accuracy))
                total_loss += loss.item()
                total_correct += bool_vector.sum()

            val_model = copy.deepcopy(model)
            val_model.eval()
            val_total_loss = 0
            val_total_correct = 0
            val_total_correct_heuristic = 0
            criterion = LabelSmoothingCrossEntropy()
            for batch_idx, (val_image, val_tags) in enumerate(val_dataloader):
                val_image = val_image.to(device)
                val_tags = val_tags.to(device)

                val_output = val_model(val_image).double()
                val_loss = criterion(val_output, val_tags)

                val_output_prob = F.softmax(val_output, dim=1)
                val_predict_vector = np.argmax(to_np(val_output_prob), axis=1)

                val_output_prob_heuristic = np.multiply(to_np(val_output_prob), heuristic_vector)
                val_predict_vector_heuristic = np.argmax(val_output_prob_heuristic, axis=1)

                val_label_vector = to_np(val_tags)
                val_bool_vector = val_predict_vector == val_label_vector
                val_bool_vector_heuristic = val_predict_vector_heuristic == val_label_vector

                val_total_loss += val_loss.item()
                val_total_correct += val_bool_vector.sum()
                val_total_correct_heuristic += val_bool_vector_heuristic.sum()

            nsml.save(epoch_idx)
            print('TRAIN: \t Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(epoch_idx,
                                                                                     args.epochs,
                                                                           total_loss / len(train_dataloader.dataset),
                                                                           total_correct / len(train_dataloader.dataset)))

            print('VAL: \t Epoch {} / {}: Loss {:2.4f} / Val Epoch Acc {:2.4f} / Heuristic Epoch Acc: {:2.4f}'.format(epoch_idx, args.epochs,
                                                                           val_total_loss / len(val_dataloader.dataset),
                                                                           val_total_correct / len(val_dataloader.dataset),
                                                                           val_total_correct_heuristic / len(val_dataloader.dataset)))

            nsml.report(summary=True, step=epoch_idx, scope=locals(), **{
                "train__Loss": total_loss / len(train_dataloader.dataset),
                "train__Accuracy": total_correct / len(train_dataloader.dataset),
                })


