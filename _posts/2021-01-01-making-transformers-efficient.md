---
title: "Making Transformers Efficient, an Introduction"
excerpt_separator: "<!--more-->"
categories:
  - Deep Learning

tags:
  - transformers
  - linformer
  - reformer
  - distilbert
  - NLP
---

## Introduction
The introduction of the transformer architecture in 2017 ([Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf)) has revolutionized the field of NLP and has slowly began influencing the field of computer vision by achieving SOTA results on Imagenet ([Dosovitskiy et al. 2020](https://arxiv.org/pdf/2010.11929.pdf)) and very recently being used by Deepmind to tackle the problem of protein folding ([Jumper et al.2021](https://www.nature.com/articles/s41586-021-03819-2.pdf)).

Transformers quickly overtook RNNs and LSTMs for every SOTA benchmark and industry for a number of reasons. Transformers are able to parallelize it's computation, as opposed to processing inputs one token at a time (sequentially) which meant that Transformers were able to take advantage of modern day deep learning GPUs,  and train on magnitudes more of data while taking significantly less time to train. Second, it's ability to parallelize its computation across multiple GPUs has lead to the creation of insanely massive language models, such as OpenAI's GPT-3, which has an absurd amount of 175 billion parameters and was trained on over 300 billion tokens (GPT-3 uses sub-word tokenization, meaning words that appear frequently get their own token, and very infrequently used words are broken up into multiple tokens). This has allowed large models to store more information in it's weights than any previous RNN or LSTM. Lastly, the transformer gives us the ability to use transfer learning to use large pre-trained models from big companies like Facebook and Google and fine-tune them on relatively small labelled datasets (only a few thousand labels needed) whereas previously with RNNs and LSTMs, there was very little flexibility for fine-tuning, and you generally needed to train a model from scratch for each separate domain.

And while the transformer architecture is impressive, we know empirically that we need more data and parameters in order to achieve SOTA performance and achieve good zero-shot classification. Large models, like GPT-3 have demonstrated amazing results in zero-shot learning when scaled directly with parameter size and data size. GPT-3 has 175 billion parameters, and assuming each weight is stored as a fp32 number, that makes the model a staggering 700 billion bytes, or 700 GBs in size to just fit a model on a GPU, not even including the extra VRAM required doing any computation. That means it would take 9 Nvidia A100s 80GB (at ~15k USD/GPU) to just fit the model on. 

While transformers have achieved unprecedented levels of performance, they've also increased the cost of computation in the order of magnitudes. For context, BERT-large came out at ~300M parameters, people thought it was too big. Fast forward 4 years later, and large models are pushing ~500B parameters, 3 magnitudes larger than BERT-large. Making transformers more efficient (in both memory and compute) makes deep learning more accessible, and enables edge computing on cpus and mobile devices.

## DistilBERT
DistilBERT ([Sanh et al. 2019](https://arxiv.org/pdf/1910.01108.pdf)), is a BERT based model that is 40% smaller than BERT-base, 60% faster, all the while retaining 97% of the performance. They were able to achieve this through a process called knowledge distillation, where a student model is trained to reproduce the behaviour of a teacher model. They do this by having 3 different loss functions: distillation loss, cosine embedding loss, and masked language modelling loss.

During normal training, we get logits after the final linear layer, which we then push through a softmax to get our probability distributions. In an ideal world, only the predicted value will be high (near 1), while the rest will be near zero. Distillation loss, defined as $$L_ce = \Sigma{t_i * log(s_i)}$$. This is the same as standard cross entropy loss, but instead of comparing our student predictions to the label, we compare it to teacher predictions, in order for the student model to better mimmic the teacher model. 

Cosine embedding loss computes the measure of the loss between two input vectors are similar or dissimilar, by using the cosine distance. The authors of the paper claimed that this helped align the direction of the student and teacher hiddens state vectors.

The final loss function is masked language modelling, which is just cross entropy loss between the student output and the label. This is the only loss function that doesn't "learn" from the teacher model.



## A brief recap of transformer self-attention

<img src="/images/making-transformers-efficient/scaled_attention.png">

In a standard transformer encoder, we use a mechanism called self-attention. We have an input embedding that goes through three different linear transformations (generally which are learnable) in order to create the query, key, and value vectors. We the do a matrix multiplication $$QK^T$$, and then scale it by $$\sqrt{d_k}$$. We then compute the softmax($$\frac{QK^T}{\sqrt{d_k}}$$) and then do one final matrix multiplication with the value vector softmax($$\frac{QK^T}{\sqrt{d_k}}V$$).


```
def self_attention(query, key, value):
    dim = key.shape[-1]
    # (Query * tranpose(key)) / sqrt(dim)
    scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(dim)
    weights = F.softmax(scores, dim = -1)
    return torch.bmm(weights, value)
```

The output tensor of a self-attention has the shape of [bs, L, L] where bs is the batch size, and L is the number of tokens in the sequence length. As you can see self-attention is $$O(n^{2})$$, where n is the sequence length. 

## Linformer
To be added

## Reformer 
To be added

