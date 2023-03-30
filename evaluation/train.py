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
import torch.utils
import torchvision.datasets as dset
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

from evaluation.model import NetworkCIFAR as Network
from search_space import utils

# don't remove this import
import search_space.genotypes

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='init learning rate min')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
SIDDD_CLASSES = 3

is_multi_gpu = False


def main():
    wandb.init(
        project="automl-gradient-based-nas",
        name=str(args.arch) + "-lr" + str(args.learning_rate),
        config=args,
        entity="automl"
    )
    wandb.config.update(args)  # adds all of the arguments as config variables

    global is_multi_gpu
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    gpus = [int(i) for i in args.gpu.split(',')]
    logging.info('gpus = %s' % gpus)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("search_space.genotypes.%s" % args.arch)
    model = Network(args.init_channels, SIDDD_CLASSES, args.layers, args.auxiliary, genotype)
    if len(gpus) > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        is_multi_gpu = True

    model.cuda()
    if args.model_path != "saved_models":
        utils.load(model, args.model_path)

    weight_params = model.module.parameters() if is_multi_gpu else model.parameters()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    wandb.run.summary["param_size"] = utils.count_parameters_in_MB(model)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        weight_params,  # model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    directory = 'SIDD'
    uid = 0
    imgs = {}
    """
    for client in os.listdir(directory):
      curr_path = f'{directory}/{client}/pcap'
      logging.info("Client structure identified")
      for subdir in os.listdir(curr_path):
        curr_path = f'{directory}/{client}/pcap/{subdir}/dataset'
        curr_type = subdir[-1:]
   
        for dayscen in os.listdir(curr_path):
          curr_path = f'{directory}/{client}/pcap/{subdir}/dataset/{dayscen}'
          logging.info("Iterate days: %s", dayscen)
          for img in os.listdir(curr_path):
            #logging.info("Add images to df")
            if dayscen == 'benign':
              imgs[uid] = {'id': uid, 'label': str(0), 'fn': img, 'path': curr_path + '/' + img}
            elif dayscen == 'malicious':
              imgs[uid] = {'id': uid, 'label': str(curr_type), 'fn': img, 'path': curr_path + '/' + img}
        uid +=1
    
    """
    #For testing purposes
    for j, client in enumerate(os.listdir(directory)):
      if j == 3:
        break
      curr_path = f'{directory}/{client}/pcap'
      logging.info("Client structure identified")
      for m, subdir in enumerate(os.listdir(curr_path)):
        if m == 10:
          break
        curr_path = f'{directory}/{client}/pcap/{subdir}/dataset'
        curr_type = subdir[-1:]
        for n, dayscen in enumerate(os.listdir(curr_path)):

          curr_path = f'{directory}/{client}/pcap/{subdir}/dataset/{dayscen}'
          logging.info("Iterate days: %s", dayscen)
          for i, img in enumerate(os.listdir(curr_path)):
            if i == 10:
              break
            if dayscen == 'benign':
              imgs[uid] = {'id': uid, 'label': str(0), 'fn': img, 'path': curr_path + '/' + img}
            elif dayscen == 'malicious':
              imgs[uid] = {'id': uid, 'label': str(curr_type), 'fn': img, 'path': curr_path + '/' + img}
            uid +=1
    
    img_df = imgs.DataFrame.from_dict(imgs,orient='index')
    img_df['label'] = img_df['label'].astype(int)
    img_df['label'] = img_df['label'].replace(3,2)
    #img_df.loc[img_df.index[(img_df['label']==3)],'label'] = 2
    logging.info("Created data frame")

    def _parse_function(filename, label):
      with open(filename, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image), dtype=torch.float32)
        image = image.permute(2, 0, 1)  # move channel dimension to second position
      return image, label
    """
    def collect_file_metadata(root_dir):
      metadata = []
      uid = 0
      for dirpath, dirnames, filenames in os.walk(root_dir):
        subdir = dirpath.split('/')[-2]
        anomaly_type = int(subdir[-1]) if subdir[-1].isdigit() else None
        for filename in filenames:
            label = 0 if 'benign' in dirpath else 1 if anomaly_type == 1 else 3 if anomaly_type == 3 else None
            file_path = os.path.join(dirpath, filename)
            metadata.append((uid, label, filename, file_path, anomaly_type))
            uid += 1
      df = pd.DataFrame(metadata, columns=['id', 'label', 'fn', 'path', 'anomaly_type'])
      df['label'] = df['label'].replace(3, 2)
      return df

    img_df = collect_file_metadata(directory)
    """
    
    file_paths = img_df.path
    file_labels = img_df["label"]

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

    #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    #valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),
                                                           eta_min=args.learning_rate_min)

    best_accuracy = 0

    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        wandb.log({"evaluation_train_acc": train_acc, 'epoch': epoch})

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        wandb.log({"evaluation_valid_acc": valid_acc, 'epoch': epoch})

        scheduler.step()

        if valid_acc > best_accuracy:
            wandb.run.summary["best_valid_accuracy"] = valid_acc
            wandb.run.summary["epoch_of_best_accuracy"] = epoch
            best_accuracy = valid_acc
            utils.save(model, os.path.join(wandb.run.dir, 'weights-best.pt'))

        utils.save(model, os.path.join(wandb.run.dir, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
    global is_multi_gpu
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        parameters = model.module.parameters() if is_multi_gpu else model.parameters()
        nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    global is_multi_gpu

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    # nn.BatchNorm layers will use their running stats (in the default mode) and nn.Dropout will be deactivated.
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
