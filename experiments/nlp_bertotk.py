import os
import argparse
import time
import datetime
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import (DataLoader, RandomSampler,
                              SequentialSampler, Dataset
                              )
from torch import nn
from transformers import (BertTokenizer, AdamW, BertConfig,
                          get_linear_schedule_with_warmup
                          )
from otk.models import BertOTK


class Dataset(Dataset):

    def __init__(self, filename, tokenizer, maxlen, test=False):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')

        self.tokenizer = tokenizer

        self.maxlen = maxlen

        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        if not(self.test):
            label = self.df.loc[index, 'label']

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()  # for BERT

        if not(self.test):
            return tokens_ids_tensor, attn_mask, label
        else:
            return tokens_ids_tensor, attn_mask


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_args():
    parser = argparse.ArgumentParser(
        description="supervised transformer + OTK layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', type=str, default='sst-2_bert_mask', help='data set for experiment')
    parser.add_argument(
        '--seed', type=int, default=1, help='random_seed')
    parser.add_argument(
        '--heads', type=int, default=1, help='number of references in OTK')
    parser.add_argument(
        '--out-size', type=int, default=3, help='number of supports in reference')
    parser.add_argument(
        '--max-iter', type=int, default=10, help='max iteration for ot kernel')
    parser.add_argument(
        '--eps', type=float, default=1., help='entropic penalty in sinkhorn')
    parser.add_argument(
        '--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument(
        '--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument(
        '--batch-size', type=int, default=32, help='batch size for supervised training')
    parser.add_argument(
        '--maxlen', type=int, default=30, help='maximum length of input sequences')
    parser.add_argument(
        '--eps-w', type=float, default=1e-8, help='adams parameter for numerical stability')
    parser.add_argument(
        '--weight-decay-w', type=float, default=0., help='adams weight decay')
    parser.add_argument(
        '--outdir', type=str, default='results/', help='where to save results')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    
    args.save_logs=False
    if args.outdir !="":
        args.save_logs=True
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
        outdir = outdir + "/bertotk"
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/otk_{}_{}_{}_{}'.format(
            args.heads, args.out_size, args.eps, args.max_iter)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/learning_{}_{}_{}_{}_{}'.format(
            args.epochs, args.lr, args.batch_size, args.eps_w,
            args.weight_decay_w)
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
    if 'sst-2' in args.dataset:
        DATASET_PATH = '/services/scratch/thoth/gmialon/dataset/sst-2/SST-2'
    elif 'rte' in args.dataset:
        DATASET_PATH = '/services/scratch/thoth/gmialon/dataset/rte/RTE'

    device = torch.device('cuda')
    pretrained_model = 'bert-base-uncased'
    #model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=2,
    #                                                      output_attentions=False, output_hidden_states=True)
    model = BertOTK(out_size=args.out_size, heads=args.heads, eps=args.eps, max_iter=args.max_iter, 
                    pretrained_model='bert-base-uncased', nclass=2, fit_bias=True, mask_zeros=True)

    model.cuda()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    train_set = Dataset(filename=os.path.join(DATASET_PATH, 'train.tsv'), tokenizer=tokenizer, maxlen=args.maxlen)
    val_set = Dataset(filename=os.path.join(DATASET_PATH, 'dev.tsv'), tokenizer=tokenizer, maxlen=args.maxlen)
    test_set = Dataset(filename=os.path.join(DATASET_PATH, 'test.tsv'), tokenizer=tokenizer, maxlen=args.maxlen, test=True)

    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=args.batch_size)
    val_loader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=args.batch_size)
    test_loader = DataLoader(test_set, sampler=SequentialSampler(val_set), batch_size=args.batch_size)

    print(len(train_set), len(val_set))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_w, eps=args.eps_w) # SST-2

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    training_stats = []
    scores = 0

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # Intialization
    model.unsup_train(train_loader)

    criterion = nn.CrossEntropyLoss()

    for epoch_i in range(0, args.epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_labels = batch[2].to(device).long()

            model.zero_grad()
            output = model(b_input_ids, b_input_mask)
            loss = criterion(output, b_labels)
            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode.
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in val_loader:

            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_labels = batch[2].to(device).long()

            with torch.no_grad():
                output = model(b_input_ids, b_input_mask)
            loss = criterion(output, b_labels)
            total_eval_loss += loss.item()
            logits = output.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(val_loader)
        if avg_val_accuracy > scores:
            scores = avg_val_accuracy

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(val_loader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    print(training_stats)

    if args.save_logs:
        print('Saving logs...')
        data = {
                'score': scores,
                'training_stats': training_stats,
                'args': args
                }
        np.save(os.path.join(args.outdir, f"seed_{args.seed}_results.npy"),
                data)
    return

if __name__ == "__main__":
    main()
