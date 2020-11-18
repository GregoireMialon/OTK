import argparse
import copy
from math import ceil
import numpy as np
import os
import torch
import torch.nn.functional as F

from loaders import load_data, load_masks
from repset.approxrepset.models import ApproxRepSet
from repset.approxrepset.utils import accuracy, AverageMeter

"""
This is the ApproxRepSet baseline using the code from the paper
http://proceedings.mlr.press/v108/skianis20a/skianis20a.pdf.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_args():
    parser = argparse.ArgumentParser(
        description="baseline approxrepset for SST-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', type=str, default='sst-2_bert_mask', choices=['sst-2_bert_mask', 'sst-2_proto']
    )
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='input batch size for training')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N',
        help='number of epochs to train')
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='learning rate for the optimizer')
    parser.add_argument(
        '--heads', type=int, default=10, help='number of heads for attention layer')
    parser.add_argument(
        '--out-size', type=int, default=20, help='number of supports for attention layer')
    parser.add_argument(
        '--dim-hidden', type=int, default=768, help='dimension of each vector')
    parser.add_argument(
        "--outdir", default="results/", type=str, help="output path")
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    # check shape

    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        outdir = args.outdir + args.dataset
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/sup"
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + f"/approxrepset"
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/learning_{}_{}_{}'.format(
            args.batch_size, args.epochs, args.lr)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/refset_{}_{}_{}'.format(
            args.heads, args.out_size, args.dim_hidden)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir

    return args

def main():
    args = load_args()
    print(args)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    errs = list()

    X_train, y_train, X_test, y_test, _ = load_data(dataset=args.dataset)
    mask_train, mask_test, _ = load_masks(args.dataset)

    X_train *= mask_train
    X_test *= mask_test

    X_train = X_train.permute(0, 2, 1).numpy()
    X_test = X_test.permute(0, 2, 1).numpy()

    y_train_ = np.zeros((y_train.size, y_train.max()+1)) # added that
    y_train_[np.arange(y_train.size),y_train] = 1 # added that
    y_train = y_train_

    y_test_ = np.zeros((y_test.size, y_test.max()+1)) # added that
    y_test_[np.arange(y_test.size),y_test] = 1 # added that
    y_test = y_test_

    n_train = int(0.8 * X_train.shape[0])
    n_val = X_train.shape[0] - n_train
    # n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    idx = np.random.permutation(n_train)
    n_train_batches = ceil(n_train / args.batch_size)
    train_batches = list()

    for i in range(n_train_batches):
        max_card = max([X_train[idx[j]].shape[1] for j in range(
            i * args.batch_size,min(( i+ 1) * args.batch_size, n_train))])
        X = np.zeros((min((i + 1) * args.batch_size, n_train) - i * args.batch_size,
                      max_card, args.dim_hidden))
        for j in range(i * args.batch_size, min((i + 1) * args.batch_size, n_train)):
            X[j - i * args.batch_size, :X_train[idx[j]].shape[1], :] = X_train[idx[j]].T
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(np.where(y_train[idx[i * args.batch_size:min((i + 1) * args.batch_size, n_train)]])[1]).to(device)
        train_batches.append((X, y))

    idx = np.random.permutation(range(n_train, X_train.shape[0]))
    n_val_batches = ceil(n_val / args.batch_size)
    val_batches = list()

    for i in range(n_val_batches):
        max_card = max([X_train[idx[j]].shape[1] for j in range(
            i * args.batch_size,min(( i+ 1) * args.batch_size, n_val))])
        X = np.zeros((min((i + 1) * args.batch_size, n_val) - i * args.batch_size,
                      max_card, args.dim_hidden))
        for j in range(i * args.batch_size, min((i + 1) * args.batch_size, n_val)):
            X[j - i * args.batch_size, :X_train[idx[j]].shape[1], :] = X_train[idx[j]].T
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(np.where(y_train[idx[i * args.batch_size:min((i + 1) * args.batch_size, n_val)]])[1]).to(device)
        val_batches.append((X, y))

    n_test_batches = ceil(n_test / args.batch_size)
    test_batches = list()

    for i in range(n_test_batches):
        max_card = max([X_test[j].shape[1] for j in range(
            i * args.batch_size, min((i+1) * args.batch_size, n_test))])
        X = np.zeros((min((i + 1) * args.batch_size, n_test) - i * args.batch_size,
                      max_card, args.dim_hidden))
        for j in range(i * args.batch_size, min((i + 1) * args.batch_size, n_test)):
            X[j - i * args.batch_size, :X_test[j].shape[1], :] = X_test[j].T
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(np.where(
            y_test[i * args.batch_size:min((i + 1) * args.batch_size, n_test)])[1]).to(device)
        test_batches.append((X, y))

    model = ApproxRepSet(args.heads, args.out_size, args.dim_hidden,
                         n_classes=2, device=device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train(X, y):
        optimizer.zero_grad()
        output = model(X)
        loss_train = F.cross_entropy(output, y)
        loss_train.backward()
        optimizer.step()
        return output, loss_train

    def test(X, y):
        output = model(X)
        loss_test = F.cross_entropy(output, y)
        return output, loss_test

    best_loss = float('inf')
    for epoch in range(args.epochs):

        model.train()
        train_loss = AverageMeter()
        train_err = AverageMeter()

        for X, y in train_batches:
            output, loss = train(X, y)

            train_loss.update(loss.item(), output.size(0))
            train_err.update(1 - accuracy(output.data, y.data), output.size(0))
        
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()

        for X, y in val_batches:
            output, loss = test(X, y)
            val_loss.update(loss.item(), output.size(0))
            val_acc.update(accuracy(output.data, y.data), output.size(0))
        
        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            best_acc = val_acc.avg
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

        print("epoch:", '%03d' % (epoch+1), "train_loss=", "{:.5f}".format(train_loss.avg),
            "train_acc=", "{:.5f}".format(1 - train_err.avg), "val_loss= {}".format(val_loss.avg),
            "val_acc= {}".format(val_acc.avg))

    model.load_state_dict(best_weights)
    print("Testing...")
    model.eval()

    test_loss = AverageMeter()
    test_err = AverageMeter()

    for X, y in test_batches:
        output, loss = test(X, y)
        
        test_loss.update(loss.item(), output.size(0))
        test_err.update(1 - accuracy(output.data, y.data), output.size(0))

    print("train_loss=", "{:.5f}".format(train_loss.avg),
          "train_acc=", "{:.5f}".format(1 - train_err.avg),
          "test_loss=", "{:.5f}".format(test_loss.avg),
          "test_acc=", "{:.5f}".format(1 - test_err.avg),
          "best_epoch", "{:.5f}".format(best_epoch)
          )
    print()

    errs.append(test_err.avg.cpu())

    print("Average accuracy:", "{:.5f}".format(1 - np.mean(errs)))

    if args.save_logs:
        print('Saving logs...')
        data = {
            'score': 1 - test_err.avg.cpu(),
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'train_acc': 1 - train_err.avg.cpu(),
            'val_score': best_acc.cpu(),
            'args': args
            }
        np.save(os.path.join(args.outdir, f"seed_{args.seed}_results.npy"),
                data)
    return

if __name__ == "__main__":
    main()
