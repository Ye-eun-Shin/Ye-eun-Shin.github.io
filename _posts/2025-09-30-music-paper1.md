---
title: "[Paper Review] Understanding Audio-Text Retrieval Through Singular Value Decomposition"
excerpt: "How to handle the challenge due to many-to-one mappings"

categories:
  - Music & Audio
tags:
  - [tag1, tag2]

permalink: /music/understanding-audio-text-retrieval/

toc: true
toc_sticky: true

date: 2025-09-03
last_modified_at: 2025-09-03
mathjax: true
---

## Understanding Audio-Text Retrieval Through Singular Value Decomposition
This is a paper of MARG lab which I'm in, accepted in SIGIR 2025. ✨  
As a first-year graduate student, I am currently exploring the research directions and publications of the senior members in my lab.  

If you are focusing on multimodal (especially audio-text) methods and the lack of dataset in contrastive learning, this paper can guide some solutions.

### 💡Background
#### Facing the Many-to-One Mapping Challenge
In this paper, **contrastive learning** is mainly used to address **Audio-Text Retrieval**.

However, many audio samples share the same descriptions (text). 🥲  
=> The diversity of (audio) representations tends to decrease.  
=> Also, contrastive learning doesn't work as well when the audio-text mapping isn't properly aligned.

#### More Effective Contrastive Learning?
Nevertheless, they struggled to find a more effective approach to contrastive learing.  
(🤔I feel that contrastive learning is almost the only option in for data scarcity & multimodal retrieval tasks like CLIP📎..)

And that's..  
**"Queue-based contrastive learning"**  
: By using this method, you can effectively leverage a large number of **negative** samples within each mini-batch.  
👍 This helps improve retrieval task performance by learning a well-aligned embedding space.  
👎 But there's still a limitation when it comes to setting **hyperparameters** (ex. queue size).


### 🎨Methods
#### 1. Contrastive Learning using Queues
> A cross-modal version of [MoCo](https://arxiv.org/abs/1911.05722)(2020) adapted for bidirectional tasks

Basically, MoCo (Queue-based contrastive learning) is used in here. However, there is a key difference in terms of **modality**.  
- MoCo -> Visual Task
- suggested method -> Audio ↔️ Text 

In MoCo, **Momentum Encoder** is used. Here's how it works:  
- In Case ) **Audio -> Text** (text matching with an audio in a text set)
    - Audio Encoder updates a momentum of query (text) encoder
    - What a momentum is... that Audio query $ θ_q $ is updating the text weights $ θ_k $ with $ θ_q $ updated by back propagation.
      - $ θ_k←m⋅θ_k+(1−m)⋅θ_q $

#### 2. Objective Function of Contrastive Learning
Using two types of loss function
1) ATC(Audio-Text Contrastive) loss
    - $ SNSIM(X, Y^+) $ (an often used loss function in contrastive learning)
    ![alt text](/assets/images/music1-1.png)
    -  Then, the loss function is wrapped once more, similar to a momentum update. This is called the <i>soft ouput</i>.



2) ATM(Audio-Text Matching) loss
    - This loss evaluates how well audio and text are aligned, typically implemented via **Cross-attention**.

#### 3. LAtent Space Embedding Rank Analysis
> To evaluate the impact of negative samples in contrastive learning 

- Applying SVD to queue embedding
  - $ E^m_{que} = [e^m_1, e^m_2, … , e^m_Q] \in R^{d*Q} $, where $ e^m_i $ denotes the **CLS momentum embedding**
  - ✅ What is **CLS embedding**?  
  You might be familiar with the **[CLS]** token..! It is the **summarized** representation of input data such as sentences, audio or multimodal data.  
  Thus, a CLS momentum embedding is the [CLS] token embedding of input sequences used in the momentum encoder.

- The rank of the queue equals to the number of normalized singular values, which is **d**!

### 🧪Experiments
They employ a BEAT-based audio encoder instead of Vit-based image encoder, which is used in the architecture called "<i>BLIP</i>".  

In these experiments, two datasets were used: 1) AudioCaps 2) Clotho.
- However, Clotho contains multiple captions for a single WAV audio file. => **One-to-many mapping**??🤔
- Interestingly, AudioCaps includes video samples. Plus, each video sample has information for audio classification, video captioning, and audio captioning.  

When training the model, the standard of selecting the best checkpoint is “$ R^{sum} $”, which sums the scores of text-to-audio (t2a) and audio-to-text (a2t) retrieval.

### 🤓The part to study
- Momentum Contrast (MoCo)
    
    The paper suggesting Queue-based contrastive learning.
    
    Studying MoCo is essential to understand this paper. 
    
- Momentum Distilation
- Audio - text dataset
    - Clotho dataset
    - AudioCaps


### ❓Questions 
- Could this kind of rank analysis method work well not only for cross-modal contrastive learning but also for general contrastive learning ??