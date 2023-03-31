import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset
import wandb
import torchvision.transforms as transforms

#Adjustments#
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
####

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

from MiLeNAS.search_algorithm.architect import Architect
from MiLeNAS.search_space import utils
from MiLeNAS.search_space.model_search import Network
from MiLeNAS.search_space.model_search_gumbel_softmax import Network_GumbelSoftmax
# don't remove this import
import MiLeNAS.search_space.genotypes

parser = argparse.ArgumentParser("sidd")
parser.add_argument('--run_id', type=int, default=0, help='running id')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=4, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--optimization', type=str, default='DARTS', help='Optimization Methods: DARTS; DARTS_V2')
parser.add_argument('--arch_search_method', type=str, default='DARTS',
                    help='Architecture Search Methods: DARTS; GDAS; DARTS_V2')
parser.add_argument('--lambda_train_regularizer', type=float, default=1, help='train regularizer parameter')
parser.add_argument('--lambda_valid_regularizer', type=float, default=1, help='validation regularizer parameter')

parser.add_argument('--early_stopping', type=int, default=0, help='early_stopping algorithm')

parser.add_argument('--group_id', type=int, default=0, help='used to classify different runs')
parser.add_argument('--w_update_times', type=int, default=1, help='w updating times for each iteration')
# parser.add_argument('--tau_max', type=float, help='initial tau')
# parser.add_argument('--tau_min', type=float, help='minimum tau')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
SIDD_CLASSES = 2

lambda_train_regularizer = args.lambda_train_regularizer
lambda_valid_regularizer = args.lambda_valid_regularizer

is_multi_gpu = False

is_wandb_used = True

def _parse_function(filename, label):
    with open(filename, 'rb') as f:
        image = transforms.functional.to_tensor(Image.open(f).convert('L'))
    image /= 255.0
    label = torch.tensor(label, dtype=torch.long)
    return image, label

def get_SIDD_data():
    #For testing purposes
    directory = 'SIDD'

    imgs = {}
    uid = 0
    label_counts = {0: 0, 1: 0}

    for client in os.listdir(directory):
        curr_path = f'{directory}/{client}/pcap'

        for subdir in os.listdir(curr_path):
            curr_path = f'{directory}/{client}/pcap/{subdir}/dataset'
            curr_type = subdir[-1:]
            if curr_type == str(1):
                for dayscen in os.listdir(curr_path):
                    curr_path = f'{directory}/{client}/pcap/{subdir}/dataset/{dayscen}'
                    for i, img in enumerate(os.listdir(curr_path)):
                        #if i == 20:
                        #   break
                        if dayscen == 'benign':
                            label = 0
                        elif dayscen == 'malicious':
                            label = 1
                        imgs[uid] = {'id': uid, 'label': str(label), 'fn': img, 'path': curr_path + '/' + img}
                        uid += 1
                        label_counts[label] += 1

    total = label_counts[1] + label_counts[0]
    neg = label_counts[0]
    pos = label_counts[1]

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)


    class_weight = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)
    #class_weight = {0: weight_for_0, 1: weight_for_1}
    #class_weight = torch.FloatTensor([class_weight[i] for i in range(len(class_weight))])

    print(label_counts)
    print(label_counts[1] / label_counts[0] * 100)
    
    img_df = pd.DataFrame.from_dict(imgs,orient='index')
    img_df['label'] = img_df['label'].astype(int)
    img_df['label'] = img_df['label'].replace(3,2)
    logging.info("Created data frame, length: %s", len(img_df.index))
    """
    def _parse_function(filename, label):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('L')  # convert to grayscale
            image = torch.tensor(np.array(image), dtype=torch.float32)
            image = image.unsqueeze(0)  # add channel dimension as the first dimension
            image /= 255
        return image, label
    """


    file_paths = img_df.path
    file_labels = torch.tensor(img_df["label"].values.reshape(-1, 1), dtype=torch.long)

    class SIDDdataset(torch.utils.data.Dataset):
      def __init__(self, file_paths, file_labels):
        self.file_paths = file_paths
        self.file_labels = file_labels
        
      def __len__(self):
        return len(self.file_paths)
    
      def __getitem__(self, idx):
        filename = self.file_paths[idx]
        label = self.file_labels[idx]
        image, label = _parse_function(filename, label)
        return image, label

    train_data = SIDDdataset(file_paths=file_paths,file_labels=file_labels)
    logging.info("Created dataset")

    return train_data, class_weight

def main():
    if is_wandb_used:
        wandb.init(
            
            name="r" + str(args.run_id) + "-e" + str(args.epochs) + "-lr" + str(args.learning_rate) + "-l(" +
                 str(args.lambda_train_regularizer) + "," + str(args.lambda_valid_regularizer) + ")",
            config=args,
            entity="team-mario"
        )
    #project="automl-gradient-based-nas",
    global is_multi_gpu

    gpus = [int(i) for i in args.gpu.split(',')]
    logging.info('gpus = %s' % gpus)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    train_data, class_weight = get_SIDD_data()


    criterion = nn.CrossEntropyLoss(weight=class_weight)
    criterion = criterion.cuda()


    # default: args.init_channels = 16, CIFAR_CLASSES = 10, args.layers = 8
    if args.arch_search_method == "DARTS":
        model = Network(args.init_channels, SIDD_CLASSES, args.layers, criterion)
    elif args.arch_search_method == "GDAS":
        model = Network_GumbelSoftmax(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    else:
        model = Network(args.init_channels, SIDD_CLASSES, args.layers, criterion)

    if len(gpus) > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        is_multi_gpu = True

    model.cuda()
    if args.model_path != "saved_models":
        utils.load(model, args.model_path)

    arch_parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
    arch_params = list(map(id, arch_parameters))

    parameters = model.module.parameters() if is_multi_gpu else model.parameters()
    weight_params = filter(lambda p: id(p) not in arch_params,
                           parameters)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    """
    optimizer = torch.optim.SGD(
        weight_params,  # model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    """
    optimizer = torch.optim.Adam(
        weight_params,  # model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    # will cost time to download the data
    #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))  # split index

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size * len(gpus),
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
        
    logging.info("Initialized training queue")
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size * len(gpus),
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)
    logging.info("Initialized validation queue")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, args)

    best_accuracy = 0
    best_accuracy_different_cnn_counts = dict()

    if is_wandb_used:
        table = wandb.Table(columns=["Epoch", "Searched Architecture"])

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj, train_loss, train_rec, train_prec, train_f1score = train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)
        logging.info('train_f1score %f', train_f1score)
        if is_wandb_used:
            wandb.log({"searching_train_acc": train_acc, "epoch": epoch})
            wandb.log({"searching_train_loss": train_loss, "epoch": epoch})
            wandb.log({"searching_train_f1score": train_f1score, "epoch": epoch})
        # validation
        with torch.no_grad():
            valid_acc, valid_obj, valid_loss, valid_f1score, valid_rec, valid_prec = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        logging.info('val_f1score %f', valid_f1score)

        scheduler.step()

        if is_wandb_used:
            wandb.log({"searching_valid_acc": valid_acc, "epoch": epoch})
            wandb.log({"searching_valid_loss": valid_loss, "epoch": epoch})
            wandb.log({"searching_valid_f1score": valid_f1score, "epoch": epoch})
            wandb.log({"search_train_valid_acc_gap": train_acc - valid_acc, "epoch": epoch})
            wandb.log({"search_train_valid_loss_gap": train_loss - valid_loss, "epoch": epoch})
            wandb.log({"search_train_valid_f1score_gap": train_f1score - valid_f1score, "epoch": epoch})

        # save the structure
        genotype, normal_cnn_count, reduce_cnn_count = model.module.genotype() if is_multi_gpu else model.genotype()
        cnn_count = normal_cnn_count + reduce_cnn_count
        if is_wandb_used:
          wandb.log({"cnn_count": cnn_count, "epoch": epoch})
        model_size = model.module.get_current_model_size() if is_multi_gpu else model.get_current_model_size()
        if is_wandb_used:
          wandb.log({"model_size": model_size, "epoch": epoch})

        # early stopping
        if args.early_stopping == 1:
            if normal_cnn_count == 6 and reduce_cnn_count == 0:
                break

        print("(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count))
        #print(F.softmax(model.module.alphas_normal if is_multi_gpu else model.alphas_normal, dim=-1))
        #print(F.softmax(model.module.alphas_reduce if is_multi_gpu else model.alphas_reduce, dim=-1))
        logging.info('search_space.genotype = %s', genotype)
        if is_wandb_used:
            wandb.log({"genotype": str(genotype)}, step=epoch-1)
            table.add_data(str(epoch), str(genotype))
            wandb.log({"Searched Architecture": table})

            # save the cnn architecture according to the CNN count
            cnn_count = normal_cnn_count*10+reduce_cnn_count
            wandb.log({"searching_cnn_count(%s)" % cnn_count: valid_f1score, "epoch": epoch})
            if cnn_count not in best_accuracy_different_cnn_counts.keys():
                best_accuracy_different_cnn_counts[cnn_count] = valid_f1score
                summary_key_cnn_structure = "best_f1_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_cnn_structure] = valid_f1score

                summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_best_cnn_structure] = epoch
            else:
                if valid_f1score > best_accuracy_different_cnn_counts[cnn_count]:
                    best_accuracy_different_cnn_counts[cnn_count] = valid_f1score
                    summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count)
                    wandb.run.summary[summary_key_cnn_structure] = valid_f1score

                    summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count)
                    wandb.run.summary[summary_key_best_cnn_structure] = epoch

            if valid_f1score > best_accuracy:
                best_accuracy = valid_f1score
                wandb.run.summary["best_valid_f1score"] = valid_f1score
                wandb.run.summary["epoch_of_best_f1score"] = epoch
                utils.save(model, os.path.join(wandb.run.dir, 'weights.pt'))


def train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    global is_multi_gpu



    rec = utils.AvgrageMeter()
    prec = utils.AvgrageMeter()
    f1 = utils.AvgrageMeter()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):

        # logging.info("epoch %d, step %d START" % (epoch, step))
        model.train()
        n = input.size(0)

        # if args.arch_search_method == "GDAS":
        #     model.set_tau(args.tau_max - epoch * 1.0 / args.epochs * (args.tau_max - args.tau_min))

        input = input.cuda()
        target = target.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        # Update architecture alpha by Adam-SGD
        # logging.info("step %d. update architecture by Adam. START" % step)
        # if args.optimization == "DARTS":
        #     architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        # else:
        architect.step_milenas(input, target, input_search, target_search, lambda_train_regularizer, lambda_valid_regularizer)

        # logging.info("step %d. update weight by SGD. START" % step)
        # w_update_times = args.w_update_times
        # len_train = len(train_queue)
        # for i_w in range(w_update_times):
        #     index = i_w + step
        #     if index - (len_train - 1) > 0:
        #         index = index - (len_train - 1)
        #     (input_w, target_w) = train_queue[index]
        #     optimizer.zero_grad()
        #     logits = model(input_w)
        #     loss = criterion(logits, target_w)
        #
        #     loss.backward()
        #     parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
        #     nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        #     optimizer.step()

        optimizer.zero_grad()
        logits = model(input)
        
        loss = criterion(logits, target)
        loss.backward()
        parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
        nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        # logging.info("step %d. update weight by SGD. FINISH\n" % step)
        recall = utils.recall(logits, target)
        precision = utils.precision(logits, target)
        f1score = utils.f1_score(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        rec.update(recall.item(), n)
        prec.update(precision.item(), n)
        f1.update(f1score.item(), n)

        # torch.cuda.empty_cache()

        if step % args.report_freq == 0:
            logging.info('train %03d Loss: %e Top1: %f Recall: %f Prec: %f F1: %f', step, objs.avg, top1.avg, rec.avg, prec.avg, f1.avg)

    return top1.avg, objs.avg, loss, rec.avg, prec.avg, f1.avg


def infer(valid_queue, model, criterion):
    global is_multi_gpu

    rec = utils.AvgrageMeter()
    prec = utils.AvgrageMeter()
    f1 = utils.AvgrageMeter()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    ba = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)
        
        recall = utils.recall(logits, target)
        precision = utils.precision(logits, target)
        f1score = utils.f1_score(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        binAcc = utils.binary_accuracy(logits, target)
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        rec.update(recall.item(), n)
        prec.update(precision.item(), n)
        f1.update(f1score.item(), n)
        ba.update(binAcc.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f BinAcc: %f %f %f %f', step, objs.avg, top1.avg, ba.avg, f1.avg, rec.avg, prec.avg)
        #print('*********** %s', f1.avg)
    return top1.avg, objs.avg, loss, f1.avg, rec.avg, prec.avg


if __name__ == '__main__':
    main()
