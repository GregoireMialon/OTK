import argparse
import numpy as np
import os
import torch
import copy

from math import ceil
from sklearn.metrics import accuracy_score,log_loss
from timeit import default_timer as timer

from loaders import load_data, load_masks
from repset.repset.models import RepSet
from repset.repset.utils import AverageMeter

"""
This is the RepSet baseline using the code from the paper
http://proceedings.mlr.press/v108/skianis20a/skianis20a.pdf.
"""

def load_args():
    parser = argparse.ArgumentParser(
        description="baseline repset for SST-2",
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
        '--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train')
    parser.add_argument(
        '--heads', type=int, default=10, help='number of heads for attention layer')
    parser.add_argument(
        '--out-size', type=int, default=20, help='number of supports for attention layer')
    parser.add_argument(
        '--dim-hidden', type=int, default=768, help='dimension of each vector')
    parser.add_argument(
        "--outdir", default="results/", type=str, help="output path")
    parser.add_argument(
        "--lr", type=float, default=0.01, help='initial learning rate')
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
        outdir = outdir + f"/repset"
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir+'/learning_{}_{}_{}'.format(
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
    np.random.seed(args.seed)
    errs = list()

    X_train, y_train, X_test, y_test, _ = load_data(dataset=args.dataset)
    mask_train, mask_test, _ = load_masks(args.dataset)

    X_train *= mask_train
    X_test *= mask_test

    X_train = X_train.permute(0, 2, 1).numpy()
    X_test = X_test.permute(0, 2, 1).numpy()

    n_train = int(0.8 * X_train.shape[0])
    n_val = X_train.shape[0] - n_train
    # n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    idx = np.random.permutation(n_train)
    n_train_batches = ceil(n_train/args.batch_size)
    train_batches = list()

    for i in range(n_train_batches):
        train_batches.append((X_train[idx[i * args.batch_size:min((i+1) * args.batch_size, n_train)]],
                              y_train[idx[i * args.batch_size:min((i+1) * args.batch_size, n_train)]]))

    idx = np.random.permutation(range(n_train, X_train.shape[0]))
    n_val_batches = ceil(n_val / args.batch_size)
    val_batches = list()

    for i in range(n_val_batches):
        val_batches.append((X_train[idx[i * args.batch_size:min((i+1) * args.batch_size, n_val)]],
                              y_train[idx[i * args.batch_size:min((i+1) * args.batch_size, n_val)]]))

    n_test_batches = ceil(n_test/args.batch_size)
    test_batches = list()
    for i in range(n_test_batches):
        test_batches.append((X_test[i*args.batch_size:min((i+1)*args.batch_size, n_test)],
                             y_test[i*args.batch_size:min((i+1)*args.batch_size, n_test)]))

    model = RepSet(args.lr, args.heads, args.out_size, args.dim_hidden, n_classes=2)

    best_loss = float('inf')
    for epoch in range(args.epochs):

        train_loss = AverageMeter()
        train_err = AverageMeter()
        
        for X, y in train_batches:
            y_ = np.zeros((y.size, y.max()+1)) # added that
            y_[np.arange(y.size),y] = 1 # added that
            y_pred = model.train(X, y_)
            train_loss.update(log_loss(y_, y_pred), n_train)
            train_err.update(1-accuracy_score(np.argmax(y_, axis=1), np.argmax(y_pred, axis=1)), y_.shape[0])

        val_loss = AverageMeter()
        val_acc = AverageMeter()

        for X, y in val_batches:
            y_ = np.zeros((y.size, y.max()+1)) # added that
            y_[np.arange(y.size),y] = 1 # added that
            y_pred = model.test(X)
            val_loss.update(log_loss(y_, y_pred), n_val)
            val_acc.update(accuracy_score(np.argmax(y_, axis=1), np.argmax(y_pred, axis=1)), y_.shape[0])

        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            best_acc = val_acc.avg
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model)

        print("epoch:", '%03d' % (epoch+1), "train_loss=", "{:.5f}".format(train_loss.avg),
            "train_acc=", "{:.5f}".format(1 - train_err.avg), "val_loss= {}".format(val_loss.avg),
            "val_acc= {}".format(val_acc.avg)
            )

    print("Testing...")
    test_loss = AverageMeter()
    test_err = AverageMeter()

    for X, y in test_batches:
        y_ = np.zeros((y.size, y.max()+1)) # added that
        y_[np.arange(y.size),y] = 1 # added that
        y_test_ = np.zeros((y_test.size, y_test.max()+1)) # added that
        y_test_[np.arange(y_test.size),y_test] = 1 # added that
        y_pred = best_model.test(X)

        test_loss.update(log_loss(y_, y_pred), y_test_.size)
        test_err.update(1-accuracy_score(np.argmax(y_, axis=1), np.argmax(y_pred, axis=1)), y_.shape[0])

    print("train_loss=", "{:.5f}".format(train_loss.avg),
          "train_acc=", "{:.5f}".format(1 - train_err.avg),
          "test_loss=", "{:.5f}".format(test_loss.avg),
          "test_acc=", "{:.5f}".format(1 - test_err.avg),
          "best_epoch=", "{:.5f}".format(best_epoch) 
        )
    print()

    errs.append(test_err.avg)

    print("Average accuracy:", "{:.5f}".format(1 - np.mean(errs)))

    if args.save_logs:
        print('Saving logs...')
        data = {
            'score': 1 - test_err.avg,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'train_acc': 1 - train_err.avg,
            'val_score': best_acc,
            'args': args
            }
        np.save(os.path.join(args.outdir, f"seed_{args.seed}_results.npy"),
                data)
    return

if __name__ == "__main__":
    main()
