# -*- coding: utf-8 -*-
import torch
from torch import nn
from .layers import OTKernel, Linear
from transformers import BertModel


class BertOTK(nn.Module):
    def __init__(self, out_size=3, heads=1, eps=0.1, max_iter=50,
                 pretrained_model='bert-base-uncased', nclass=2,
                 fit_bias=True, mask_zeros=True):
        super().__init__()
        self.transformer = BertModel.from_pretrained(pretrained_model)
        self.attention = OTKernel(768, out_size, heads=heads,
                                  eps=eps, max_iter=max_iter)
        self.out_features = out_size * heads * 768 
        self.nclass = nclass
        self.classifier = Linear(self.out_features, nclass, bias=fit_bias)
        self.mask_zeros = mask_zeros

    def forward(self, input_ids, input_mask):
        # assumes that the dataset has been tokenized before
        outputs = self.transformer(input_ids, token_type_ids=None,
                                   attention_mask=input_mask)
        output = self.attention(outputs.last_hidden_state,
                                input_mask > 0).reshape(
                                outputs.last_hidden_state.shape[0], -1)
        return self.classifier(output)

    def unsup_train(self, data_loader, n_samples=5000, wb=False,
                    use_cuda=True):
        cur_samples = 0
        print("Training attention layer")
        for i, batch in enumerate(data_loader):
            data, masks, _ = batch
            if cur_samples >= n_samples:
                continue
            if use_cuda:
                data = data.cuda()
                masks = masks.cuda()
            with torch.no_grad():
                # data = self.ckn_representation(data)
                outputs = self.transformer(data, token_type_ids=None,
                                           attention_mask=masks)
                data = outputs.last_hidden_state
            if i == 0:
                patches = torch.empty([n_samples]+list(data.shape[1:]))

            size = data.shape[0]
            if cur_samples + size > n_samples:
                size = n_samples - cur_samples
                data = data[:size]
            patches[cur_samples: cur_samples + size] = data
            cur_samples += size
        print(patches.shape)
        self.attention.unsup_train(patches, wb=wb, use_cuda=use_cuda)
       

class BertMeanPool(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', nclass=2,
                 fit_bias=True, mask_zeros=True, num_out=30*768, 
                 n_hidden=30*768/2):
        super().__init__()
        self.transformer = BertModel.from_pretrained(pretrained_model)
        self.nclass = nclass
        if n_hidden:
            self.classifier = nn.Sequential(
                nn.Linear(num_out, n_hidden, bias=fit_bias),
                nn.ReLU(inplace=True),
                nn.Linear(n_hidden, nclass, bias=fit_bias))
        else:
            self.classifier = nn.Linear(num_out, nclass, bias=fit_bias)
        self.mask_zeros = mask_zeros

    def forward(self, input_ids, input_mask):
        # assumes that the dataset has been tokenized before
        outputs = self.transformer(input_ids, token_type_ids=None,
                                   attention_mask=input_mask)
        # the [CLS] token is included in the pooling
        if self.mask_zeros:
            output = outputs.last_hidden_state * input_mask.unsqueeze(-1)
        else:
            output = outputs.last_hidden_state
        return self.classifier(output.reshape(output.shape[0], -1))
 
