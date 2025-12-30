---
title: "[Paper Review] L3AC: Towards a Lightweight and Lossless Audio Codec"
excerpt: "Neural Audio Codec : Single Quantizer Wins!"

categories:
  - Music & Audio
tags:
  - [tag1, tag2]

permalink: /music/l3ac/

toc: true
toc_sticky: true

date: 2025-12-30
last_modified_at: 2025-12-30
mathjax: true
---
# Why This Paper Matters to My Research 😍
I came across this paper while working on my own research on **real-time music generation**, and it immediately resonated with several challenges I was facing 🧐.

I need **an optimal neural audio codec** for music generation that **supports real-time inference**.

For raw-audio streaming generation, codec design must be evaluated along **four key axes** :  
- **Music suitability** : Whether the codec is trained on and sup-
ports music audio. Codecs optimized for speech or speech-
dominant datasets often fail to capture musical structure.

- **Causality and receptive field** : Whether the codec operates causally, and if not, the extent of its receptive field. Any non-causal computation introduces a receptive field that extends into the future, incurring latency proportional to the look-ahead window.

- **Computational efficiency**: The runtime cost of encoding
and decoding, which affects overall real-time responsiveness.
- **Codebook depth**: Deeper codebooks increase token rates and generation cost, while shallower codebooks favor low-latency generation by prioritizing coarse musical attributes critical for interaction.

And.. I found that L3AC of this paper should be the solution :)


# Significance 🎯

## Single Quantizer Wins!

Many state-of-the-art (SOTA) neural codecs rely on complex, **multi-quantizer** architectures to achive their high performance in **compression** and **reconstruction**.

These systems typically employ multiple quantizers arranged hierarchically, each capturing different levels of detail in the audio signal.

But.. here’s the catch — **stacking more quantizers doesn’t always mean better performance.**

![The star point is the L3AC model.](/assets\images\music2-1.png)

The star point is the L3AC model.

Interestingly, as shown in Figure 1, **WavTokenizer** (Ji et al. 2025), **TAAE** (Parker etal. 2025), **UniCodec** (Jiang et al. 2025) and the star point codec (**L3AC**, Zhai et al. 2025) are all **single-quantizer** audio codecs. They achieves a higher **STOI** than multi-quantizer ones generally!

Also, there are **cons** when using multi-quantizer architectures.

1. The hierachical token streams necessitate **customized aggregation techniques** to support downstream models.
    
    ```yaml
    t=0 → [12,  3, 55,  9]   # level 0~3        
    t=1 → [87, 14, 22, 31]
    ...
    
    => (**aggregation**) => [t_0, t_1, t_2, ...]
    ```
    
2. **Inconsistencies** can emerge in the generated tokens. These inconsistencies make it challenging for langauge models to predict subsequent tokens.
    
    🤔 I guess the inconsistency the authors mention here refers to **temporal inconsistency** or **cross-level semantic inconsistency**.
    

## One Quantizer to Rule Them All? Not Yet.

**✅ High objective fidelity**                         ❌ WavTokenizer (Ji et al. 2025), UniCodec (Jiang et al. 2025)

**✅ Cross-domain generalizability**          ❌ SingleCodec (Li et al. 2024), TAAE (Parker et al. 2025)

**✅ Computational efficiency**

⇒ Only L3AC audio codec checks all the boxes! :)

# Main Idea 💡

> L3AC is a lightweight and efficient nerual audio codec that delivers high-fidelity audio with low computational complexity and strong scalability.
> 

## TConv (TPooling)  & Local Transformer
This paper achieves a sufficient receptive field to capture both 1️⃣**fine-grained details** and 2️⃣**long-range signal dependencies**.  
For receptive field expansion, authors propose a hybrid architecture that combines **convolutional layers with local transformers.**

Although local transformers capture the global context at acoustic level, authors argue that **shallow convolutional layers still struggle to model longer signal-level patterns**, <i> e.g. the 10ms trends</i> and **dominant short-term variants overshadow the long-term trends** in the periodic nature of audio signals.  
(L3AC uses shallow conv layers because deeper layers introduces considerable computational overhead.)

To address this problem, L3AC introduces **TPooling**, a novel pooling structure designed to explicitly **capture global amplitude variations**.  
**TPooling** is defined as:  
$TPooling(x,K)=AvgP(MaxP(|x|,K),K)$

If you read this section, you might be curious how they claim that **TPooling capture longer signal-level patterns**. 
And here is the answer of this..
![alt text](/assets/images/music2-8.png)
I think the red line is like.. **the spectral envelope**.. 🤔
## Architecture
![alt text](/assets/images/music2-2.png)
**EnCodec**-based Neural Codec‼️

| **features** | **EnCodec (Meta)** | **L3AC (Zhai et al)** |
| --- | --- | --- |
| **Base Architecture** | SEANet (Conv1d + Snake Activation) | SEANet (Conv1d + Snake Activation) |
| **Sequence Modeling** | **LSTM** (2 layers) | **Local Transformer** (Local Attention) |
| **Quantization** | **RVQ** (Residual Vector Quantization) | **FSQ** (Finite Scalar Quantization) |
| **purpose** | High-fidelity (using multiple codebooks) | Lightweight compression with a single quantizer |

**RVQ vs FSQ ❓**  
**Residual Vector Quantization** represents latent features using **a sequence of codebooks**, enabling high-fidelity reconstruction at the cost of increased latency.
```bash
x → Q1 → r1 # across the first layer quantizer
r1 → Q2 → r2
r2 → Q3 → ...
```

**Finite Scalar Quantization** performs **independent** scalar quantization per dimension, offering a lightweight and low-latency alternative suitable for real-time applications.  
```bash
z_t = [z₁, z₂, z₃, z₄]
↓
[Q(z₁), Q(z₂), Q(z₃), Q(z₄)]
```

# Experiments 🔍

## Settings

- datasets
    
    
    | domain | datasets |
    | --- | --- |
    | Speech | **LibriSpeech (Panay-otov et al. 2015)** for clean speech & **Common Voice (Mozilla 2024)** for noisy speech |
    | Music | The low-quality version of MTG-Jamendo dataset (Bogdanov et al. 2019) |
    | General Audio | The FSD50K dataset (Fonseca et al. 2022) |
- Training Details
    - AdamW optimizer + one-cycle lerarning rate schedule (Smith and Topin. 2019)
    - Gradient clipping
    - a weight decay of 1*10^(-5)
    - **a single NVIDIA RTX 4090 GPU ⇒ demonstrating the model’s computational efficiency**

## Results
### Signal-level Evaluation 
![alt text](/assets/images/music2-5.png)
This evaluation was conducted using the **Codec-SUPERB benchmark (Wu et al. 2024)**, which is contained only speech-domain dataset.
### Application-level Evaluation
![alt text](/assets/images/music2-7.png)
### Generated Tokens Evaluation 
![alt text](/assets/images/music2-6.png)

# My Experiments 👩🏻‍🔬
## L3AC Causality and Receptive Field Analysis
> <i>The downsampled features are fed into a Local Transformer, which provides a large acoustic-level  receptive field with low computational overhead and latency. To maintain causality and ensure real-time performance, the transformer avoids backward dependencies.</i>  

The paper claims **causality** due to the use of a **local Transformer**, and I empirically verify this claim 🤓.

- Reference  
    👾train code: [https://github.com/zhai-lw/L3AC/blob/train/src/model/exp/train.py](https://github.com/zhai-lw/L3AC/blob/train/src/model/exp/train.py)
        
    - Model architecture `EnCodec`
        
        [https://github.com/zhai-lw/L3AC/blob/main/l3ac/en_codec.py#L22](https://github.com/zhai-lw/L3AC/blob/main/l3ac/en_codec.py#L22)

- **Model Info (3kbps)**
    | feature | num |
    | --- | --- |
    | **bitrate** | **3kbps** |
    | sampling rate | 16,000 |
    | params | 10.31M |
    | codebook_size | 250047 |
    | frame_rate | 59.2…. |
    | compress_rate (downsampling) | 64 |
    | feature_dim | 128 |

### Results
1️⃣ $d y[t] / d x$  
<i>Which part of the input signal contributes to the generation of the reconstructed output at time step t?</i>  

When **81,920 samples** is the input length, the length of look-ahead is **100ms (= 1600 samples)**

![image.png](/assets/images/music2-3.png)

**Interpretation**  
Due to the **strong asymmetry between past and future context (past ≫ future)**, the model can be considered to **rely almost entirely on past information**.

**t <= 10,000 (past direction):**  
Gradients remain significant over a wide temporal range spanning several thousand samples.  
In particular, a strong peak is observed around the region from t − 2000 to t, indicating heavy reliance on recent past context.

**t > 10,000 (future direction):**  
Gradients rapidly diminish and become nearly zero beyond approximately 125 ms (1,000 samples).  
This suggests that the model is not strictly causal, but rather follows a near-causal structure based on a local transformer, attending to a small local future context.

➡️ L3AC is **not strictly causal.**  
Plus, it turns out that there is no fixed latency between the input and the target.

2️⃣ $d z_q[t] / d x$  
<i>Which part of the input signal contributes to the generation of the token at time step t?</i>  

    Target Frame (Quantized): 103
    target sample : 10000
    Receptive Field Range (Input Samples): 0 ~ 10746
    Receptive Field Length: 10747 samples
    Receptive Field Duration: 0.6717 seconds
    
![image.png](/assets/images/music2-4.png)
    
➡️ L3AC has a look-ahead of approximately **750 samples (~47 ms)** when generating a single token.


# Conclusion
This paper provides a solid and well-organized overview of recent neural audio codecs, which makes it a valuable reference for understanding the current landscape. 🤩

However, the evaluation results on **discrete tokens** are somewhat **disappointing**🥲. This raises concerns that performance may further degrade when these tokens are used as inputs to a language model for **downstream generation tasks**.  
→ In my own inference experiments, **the reconstruction quality on music data** was noticeably **inferior to DAC**, often exhibiting significant artifacts and noise.

Another limitation is that the evaluation primarily focuses on **speech-related tasks**, leaving its effectiveness on music-oriented scenarios less explored.🥲

Overall, while the method shows promise as an efficient and flexible neural audio codec, its effectiveness for 🎵**music generation**🎵 remains an open question. 