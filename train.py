import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from configs.load_configs import load_configs
from datasets.Flowers import Flowers
from tools.utils import remove_dir_and_create_dir, create_model, set_seed, model_parallel



def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    weight_dir = args.summary_dir + '/weights'
    log_dir = args.summary_dir + '/logs'
    remove_dir_and_create_dir(weight_dir)
    remove_dir_and_create_dir(log_dir)
    writer = SummaryWriter(log_dir)

    set_seed(args.seed)

    nw = args.num_workers
    print('Using {} dataloader workers every process'.format(nw))

    Train = Flowers(args.dataset_train_dir, args.batch_size, nw, aug=True)
    train_loader, train_dataset = Train.loader, Train.dataset

    Val = Flowers(args.dataset_val_dir, args.batch_size, nw)
    val_loader, val_dataset = Val.loader, Val.dataset

    train_num, val_num = len(train_dataset), len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model = create_model(args=args)


    model = model_parallel(args, model)
    model.to(device)

    if args.weights != '':
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict))

    loss_function = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-5)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf 
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0

    for epoch in range(args.epochs):
        
        model.train()
        train_acc = 0
        train_loss = []
        train_bar = tqdm(train_loader)

        for data in train_bar:
            train_bar.set_description("epoch {}".format(epoch))
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()
            out = model(images)
            prediction = torch.max(out, dim=1)[1]

            loss = loss_function(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))
            train_acc += torch.eq(labels, prediction).sum()

            del images, labels

        model.eval()
        val_acc = 0
        val_loss = []
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                out = model(images)
                loss = loss_function(out, labels)
                prediction = torch.max(out, dim=1)[1]

                val_loss.append(loss.item())
                val_acc += torch.eq(labels, prediction).sum()

                del images, labels

        val_accurate = val_acc / val_num
        train_accurate = train_acc / train_num
        print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
              format(np.mean(train_loss), train_accurate, np.mean(val_loss), val_accurate))

        writer.add_scalar("train_loss", np.mean(train_loss), epoch)
        writer.add_scalar("train_acc", train_accurate, epoch)
        writer.add_scalar("val_loss", np.mean(val_loss), epoch)
        writer.add_scalar("val_acc", val_accurate, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), "{}/epoch={}_val_acc={:.4f}.pth".format(weight_dir,
                                                                                   epoch,
                                                                                   val_accurate))











if __name__ == '__main__':

    main(load_configs())