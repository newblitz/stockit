# Integrative stock price trend prediction via hierarchical LLM text processing and patch-based transformer with co-attention

**Yuntao Zhang ᵃ,\*, Zheng Dong ᵇ,¹, Wenrui Xu ᶜ,¹**

ᵃ *Nanjing Normal University, Department of Electrical Engineering and Automation, Nanjing, Jiangsu, 210046, China*
ᵇ *Beijing Bytedance Technology Co Ltd, 48 Zhichun Road, Beijing, Beijing, 100098, China*
ᶜ *CNIC Corporation Limited, Tower A Jinjia Plaza No 6 Financial Street, Beijing, Beijing, 100033, China*

\* Corresponding author.
E-mail addresses: 231802008@njnu.edu.cn (Y. Zhang), zhdong0@outlook.com (Z. Dong), xuwenrui2023@outlook.com (W. Xu).
¹ Due to the absence of institutional email addresses provided by their employer, authors Zheng Dong and Wenrui Xu are unable to provide such addresses. Personal email addresses are included in the manuscript for contact purposes.

Journal: *Expert Systems With Applications* 302 (2026) 130441
DOI: https://doi.org/10.1016/j.eswa.2025.130441
Received 19 July 2025; Received in revised form 18 October 2025; Accepted 12 November 2025; Available online 19 November 2025
0957-4174/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

---

## Article Info

**Keywords:** Stock movement prediction; Fusion model; Attention mechanism; Large language models; Patching technique; Multimodal framework

## Abstract

Recent advances in large language models have improved semantic extraction; however, existing approaches still struggle to distill truly market-moving signals from lengthy, redundant financial texts, and simple concatenation or one-way cross-attention often fails to capture the bidirectional dependencies between textual semantics and price dynamics. We propose a multimodal framework with hierarchical summarization and co-attention that jointly learns from financial news and historical prices for stock trend prediction. The framework comprises three components. A hierarchical LLM progressively refines text, removing redundancy and extracting multi-level cues most relevant to market movements. A patch-based, single-channel Transformer models price sequences and text streams in parallel, reducing attention complexity while preserving each modality's temporal and contextual structure. A co-attention mechanism dynamically aligns and reinforces cross-modal features to yield robust joint representations. In experiments, our method achieves 67.01% directional accuracy with a Matthews Correlation Coefficient of 0.346 on CMIN-US and 70.24% accuracy with a Matthews Correlation Coefficient of 0.293 on CMIN-CN. Ablation results confirm that combining hierarchical summarization with co-attention substantially improves multimodal forecasting performance.

---

## 1. Introduction

Artificial Intelligence (AI) has become indispensable in the financial sector, powering critical applications such as stock price trend forecasting (Long et al., 2020; Zhao & Yang, 2023), robo-advisory services (Bertrand et al., 2023; Rossi & Utkus, 2024), and risk management (Ahbali et al., 2022; Wang et al., 2021). Traditional approaches relying on historical price series and classical statistical models (e.g., SVM, WNN) (Huang et al., 2005; Lei, 2018) often fail to capture fast-moving market dynamics driven by unstructured information. Consequently, integrating alternative data sources — financial news, social-media sentiment, and corporate disclosures — into forecasting models has gained prominence, yet effectively extracting and fusing these heterogeneous signals remains a major challenge (Li et al., 2014).

The use of textual information in financial markets has become increasingly significant. Text data, including financial news, analysis reports, and social media comments, plays a pivotal role in reflecting market sentiment and revealing fundamental information. This information not only provides investors with sentiment and public opinion insights but also offers a new dimension of data for stock trend prediction. In recent years, numerous studies have explored how text data can enhance the accuracy of stock price predictions. For example, several studies have integrated text information into stock prediction models: Ruan et al. (2018) employ sentiment analysis tools to reveal correlations between the information in tweets and stock returns; Nam and Seong (2019) use causal analysis to link financial news to stock price fluctuations in the Korean stock market; Li et al. (2021) propose a multimodal LSTM model that enables interaction between online news text and stock price data. Additionally, Lin et al. (2022) explore the influence of text feature representations, machine learning models, and news platforms on text-mining-based stock prediction.

Despite the significant potential of textual information in stock prediction, the challenge of precisely extracting key information from vast and diverse financial texts remains a major hurdle in the fintech domain. To address this, many studies have applied Natural Language Processing (NLP) techniques to analyze financial texts. For example, Sawhney et al. (2020) use BERT to deeply represent social media text and combine it with company correlation data to enhance stock trend prediction performance; Jain and Agrawal (2024) combine BERT with Generative Adversarial Networks (GAN) to predict stock prices; and Euna et al. (2020) combine BERT sentiment analysis results with macroeconomic indicators to predict stock prices. Although pre-trained models such as BERT show great potential in text understanding, traditional NLP techniques still exhibit significant limitations when processing vast and complex financial texts. Current methods fail to capture the deep semantics and subtle contextual relationships within texts, which substantially limits the accurate extraction and utilization of key information.

The rise of Large Language Models (LLMs) in recent years has brought revolutionary advancements to NLP, profoundly reshaping the landscape of natural language processing. The emergence of LLMs has made it possible to extract fine-grained insights from large amounts of unstructured text, thus providing new opportunities for improving the precision and adaptability of stock prediction models. For instance, Chiu et al. (2025) use ChatGPT to analyze financial news articles and assign a buy score to each stock mentioned, quantifying the future trend of that stock; Wang et al. (2024b) combine pre-trained LLMs with Sequential Knowledge-Guided Prompting (SKGP) to automatically identify factors influencing stock volatility, using historical stock price data in textual format to predict stock movements; Li et al. (2024a) use pre-trained LLMs to build a denoised news encoder, extracting valuable information from financial news to improve multi-stock movement prediction. However, the characteristics of stock data present unique challenges for the application of LLMs. Firstly, stock-related data typically encompass extensive textual information, including financial news, corporate financial reports, social media commentary, and analyst reports. These texts are often high-dimensional, heterogeneous, and time-sensitive, requiring models that can process and extract key information rapidly (Boudoukh et al., 2013). Secondly, existing LLMs usually impose input length limitations — for instance, the mT5 model is limited to 512 tokens (Xue et al., 2021). Moreover, since these models are built on the Transformer architecture (Vaswani et al., 2017; Yang et al., 2023) with a computational complexity of O(n²), the resource cost increases substantially when processing long sequences. Finally, long texts may prevent the model from accurately focusing on critical time points or events, thereby affecting prediction performance (Liu et al., 2024a).

To address the issues described above and leverage the strengths of LLMs in text comprehension, we propose an enhanced approach. Specifically, we employ a pre-trained, task-specific LLM and introduce a hierarchical summarization technique to extract key information from financial texts (see Section 3). In our method, the original text is segmented into discrete blocks that are iteratively refined. This process not only enables a deep capture of intrinsic semantics and subtle contextual interrelations — thereby accurately identifying the critical factors affecting stock volatility — but also mitigates the limitations of traditional approaches in precise information extraction.

Once textual information is mined using LLMs, the next challenge lies in fusing multimodal data for predictive tasks. Traditional fusion methods, such as simple feature concatenation or weighted sums, typically combine different types of data in a straightforward manner with minimal additional parameter tuning. For example, Xu and Cohen (2018) concatenate Twitter sentiment features with historical price features and input them into the model to predict stock movements. Xu et al. (2020) apply predefined weights to combine tweet data and price data to predict stock price changes. While these simple fusion methods are convenient, they have significant limitations: they fail to adequately capture the complex relationships and potential interactions between different modalities, thus failing to fully leverage the complementary advantages of each modality, which limits the model's performance and flexibility. To address this, attention-based fusion methods have been introduced. For example, Zhang et al. (2022) incorporate a Transformer self-attention mechanism to dynamically allocate weights when fusing news text and stock price data; Zhang et al. (2024c) use attention strategies to balance the importance of stock prices and text data, improving prediction accuracy; Gao et al. (2021) implement a time-aware relational attention network to fuse multiple modalities of stock data, including prices and news. Although these attention-based methods are capable of dynamically assessing the contribution of each modality to the prediction task and capturing complex interdependencies and potential interactions — thereby significantly enhancing multimodal data fusion and model accuracy — they fall short in adequately capturing local features and suffer from computational inefficiencies.

To address the aforementioned issues, this paper adopts a single-sided Transformer architecture combined with patch technology to model both text and price data and proposes a novel text-price co-attention module (Section 3). This module simultaneously focuses on textual and price features, capturing the deep interrelations and complementary information between the two modalities. In doing so, it overcomes the limitations of traditional attention mechanisms in local feature extraction and computational efficiency.

### 1.1. Objectives and contributions

In summary, addressing the challenges of integrating heterogeneous financial news and historical prices for trend forecasting, this work delivers the following objectives and contributions:

- A novel multimodal stock prediction framework is proposed to perform trend forecasting based on diverse financial data, addressing the challenges of parallel data processing, feature fusion, and joint representation learning. The framework's components collaborate to extract textual and price features, fuse them, and achieve precise predictions.
- The semantic extraction capability of the LLM is leveraged for multi-level information refinement, the removal of redundant content, and the accurate capture of key semantic cues driving stock price fluctuations.
- A patch-based single-channel Transformer architecture is introduced to model stock price sequences and textual streams in parallel, preserving each modality's temporal and contextual structures while enhancing the efficiency of attention computations.
- A co-attention fusion mechanism is developed to accommodate the heterogeneity of text and historical price data, dynamically aligning and mutually reinforcing multimodal features to further enhance prediction accuracy.

The remainder of the paper is organized as follows. Section 2 reviews the application of text data in stock trend prediction and the latest advances of LLMs in temporal modeling. Section 3 presents a detailed description of the overall model architecture, including the hierarchical summarization process, the patch-based single-channel Transformer modeling for both stock price and text data, and the implementation details of the text-price co-attention module. Section 4 describes the experimental design, datasets, and evaluation metrics, followed by an in-depth analysis of the experimental results. Finally, Section 5 concludes the paper and discusses future research directions.

---

## 2. Related work

The application of artificial intelligence (AI) in equity trading has emerged as a prominent research area, encompassing portfolio optimization, market forecasting, sentiment analysis of financial news, and hybrid frameworks that integrate multiple methodologies (Ferreira et al., 2021; Shen et al., 2021; Ye et al., 2023). In particular, AI-driven stock trend prediction has achieved state-of-the-art performance (Jiang et al., 2023; Liu et al., 2023; Tang et al., 2019). In this section, we review recent advances in two key areas: stock trend prediction and deep learning for time-series forecasting.

### 2.1. Stock trend prediction

Forecasting stock price trends remains a fundamental challenge due to the inherent complexity, randomness, and dynamic evolution of financial markets. To better exploit heterogeneous and unstructured data sources, recent research has focused on three complementary directions:

**Multimodal information processing.** Recent studies demonstrate that fusing heterogeneous modalities — such as financial news text, sentiment scores, and numerical indicators — yields significant gains in stock-trend forecasting. Luo et al. (2023a) combine financial news text with a causality-augmented matrix to capture cross-modal interactions for trend forecasting. Ma et al. (2023a) design a multi-source aggregation classifier that fuses numerical market indicators with news-driven sentiment scores. Wang et al. (2020) propose a scenario-detection framework to extract bullish and bearish expert viewpoints, further enhancing prediction accuracy. Wang et al. (2024a) combine long short-term memory (LSTM) networks with a Transformer to model minute-level price series alongside investor sentiment and COVID-19 indicators in the pharmaceutical sector, reporting superior accuracy, F1-score, precision, and recall, and uncovering strong causal links via Granger-causality and impulse-response analyses. Anbaee Farimani et al. (2024) propose an adaptive multimodal framework that addresses non-stationarity in financial time series by employing recurrent convolutional neural networks (RCNNs) and RNN layers to capture complex temporal dependencies in news, sentiment, and technical indicators — demonstrating robust performance across different markets. Liu et al. (2024b) introduce Melody-GCN, a multiscale multimodal dynamic graph convolution network that progressively refines temporal features and aligns numerical and textual representations, outperforming state-of-the-art baselines in live trading simulations. Finally, Wang et al. (2023) apply tensor robust PCA (TRPCA) to organically fuse multimodal, multitemporal matrices and feed the result into an attention-augmented LSTM, achieving further accuracy gains; however, high-frequency components and distributed anomalies can distort the decomposition, limiting TRPCA's robustness.

**Feature selection.** Effective feature selection remains critical for improving stock-market forecasts by identifying and retaining the most informative predictors. Haq et al. (2021) introduce a multi-filter selection pipeline coupled with a deep generative model to improve future price movement forecasts. Chaudhari and Thakkar (2023) embed a coefficient-of-variation selector within a neural network to reduce feature redundancy and boost performance. Furthermore, Zhen et al. (2025) apply five distinct selection algorithms to hundreds of trading features, integrate three investor-sentiment indicators, and feed the refined feature set into multiple deep-learning models, achieving significant gains. Htun et al. (2023) survey 32 studies (2011–2022), highlighting correlation-based criteria, random forests, principal component analysis, and autoencoders as the most widely employed and effective methods. Han et al. (2023a) address the sensitivity of traditional up-down labeling by proposing N-period Min-Max (NPMM) labeling combined with XGBoost, which significantly enhances model efficiency and accuracy. Dezhkam and Manzuri (2023) integrate the Hilbert-Huang transform with XGBoost to boost portfolio optimization performance, while Maqbool et al. (2023) fuse sentiment analysis from financial news with machine learning, achieving up to 0.90 accuracy in short-term trend prediction. These works confirm that careful selection of a small, high-impact feature subset reduces computational cost and mitigates overfitting, thereby enhancing both performance and interpretability.

**Model architectures.** Graph-based neural models have gained traction for capturing complex inter-stock interactions and temporal dependencies. Chen et al. (2021) propose GC-CNN, which extracts graph-convolutional features from a stock-correlation graph for subsequent CNN-based trend prediction. Liu et al. (2024c) present Melody-GCN, a multimodal, multiscale dynamic graph convolutional network that aligns numerical and textual modalities, extracts temporal representations at multiple scales, and learns evolving spatiotemporal relationships. Ma et al. (2024) generate multiple related graphs from historical data and establish VGC-GAN, a multi-graph convolutional adversarial framework to enhance forecasting robustness. Gregnanin et al. (2025) propose a dynamic graph network that integrates attention-enhanced LSTM to model geometric and temporal patterns in price series, significantly outperforming traditional approaches. Pang et al. (2023) develop PriceExploration-Network (PE-Net), which constructs dynamic price graphs via clustering and extracts spatiotemporal embeddings without external data, demonstrating the efficacy of purely price-based relations. Ansari (2024) present Multi-Cluster Graph (MCG) GNNs, where multiple clustering strategies yield multi-relation graphs fed into LSTM, yielding marked accuracy improvements. Tian et al. (2024) introduce a dynamic hypergraph spatio-temporal network (DHSTN) that uses GRU and hypergraph attention to learn evolving higher-order stock relations. Although powerful, these graph approaches incur quadratic computation in nodes and edges, and require precise temporal alignment between graph structure and time-series inputs.

These approaches — from multimodal fusion and advanced feature selection to graph-structured modeling — have substantially advanced our ability to extract predictive signals from structured and unstructured financial data (Dong et al., 2021; Zheng et al., 2017). Nonetheless, challenges remain in jointly modeling fine-grained temporal dynamics, maintaining computational efficiency, and ensuring robust multimodal alignment.

### 2.2. Deep learning for time-series forecasting

In multivariate time-series forecasting, recent work has concentrated on three main directions:

**Attention-based encoder-decoder architectures.** Attention-augmented encoder-decoder models enable adaptive focus on the most informative input segments and dynamic context selection during decoding, yielding enhanced long-range dependency modeling. For example, Das et al. (2023) propose TiDE, an MLP-based encoder-decoder model that combines the simplicity of linear models with the capacity to capture nonlinear covariates. Du et al. (2020) employ a Bi-LSTM encoder-decoder framework that jointly learns context and temporal attention vectors to adaptively model long-term dependencies and hidden correlations. Nguyen and Quanz (2021) introduce a probabilistic temporal latent auto-encoder that performs end-to-end nonlinear decomposition, achieving up to 50% improvement over benchmarks. Bi et al. (2024) propose VBAED, which couples variational mode decomposition (VMD) with bidirectional input and temporal attention in BiLSTM blocks, achieving state-of-the-art water-quality predictions. Wu and Zhang (2023) develop an attention-based encoder-decoder for lithium-ion battery state-of-charge estimation under varying temperatures, reducing MAE to within 0.77%. Huang et al. (2023) integrate dual-stage attention into an LSTM encoder-decoder for high-arch dam displacement forecasting, surpassing both regression and standard neural methods. Fu et al. (2024) introduce FGNet, embedding novel Fourier attention into the self-attention encoder for chaotic multi-step time-series prediction, demonstrating strong empirical performance.

**PatchTST and hybrid models.** "Patch" techniques divide long sequences into shorter, learnable windows, mitigating catastrophic forgetting and redundant noise. To reduce computational cost and better capture local patterns, a series of PatchTST-based methods have been developed. Ge et al. (2024), Zhang et al. (2024a), Zheng et al. (2025). Zhang et al. (2025b) present PatchTCN, a hybrid Transformer-CNN model integrating InterPatch-Conv and MSPatch-Conv modules. Li et al. (2025) extend PatchTST to ECG anomaly detection by incorporating subtractive attention and data augmentation. Qu et al. (2025) combine seasonal-aware PatchTST reconstruction, variational mode decomposition (VMD), and Progressive Layered Extraction (PLE) multi-task learning for ultra-short-term multivariate load forecasting. Cao et al. (2025) further propose MSPatch, a multi-scale patch-mixing framework that enhances long-term forecasting performance. Zheng et al. (2025) propose MPFGSN, a multi-resolution patch-based Fourier graph spectral network that captures multi-scale spatiotemporal correlations more efficiently than traditional graph convolutions. Yu et al. (2025) present SPMST, a symmetric patch-mask Siamese Transformer for few-shot multivariate anomaly detection, matching larger-data baselines. PVMTF frameworks apply patch partitioning with GRU-based gating to mid-term photovoltaic power forecasting, preserving historical information while reducing complexity. PatchTST models further excel at one-month ocean-wave height forecasting, achieving superior COR, RMSE, and MAE over Informer, LSTM, and NeuralProphet.

**Large language models for time series.** LLMs have been leveraged both to generate high-quality summaries and to enrich time-series representations. Inspired by their success in NLP, LLMs have been applied to time-series forecasting. Hu et al. (2025) combine pre-trained LLMs with patch reprogramming for prediction. Others encode time series as sequences of tokens, leveraging LLMs' capacity to model multimodal distributions and handle missing data via textual placeholders, enabling zero-shot inference (Gruver et al., 2023). Moreover, LLMs have been used to extract insights from both textual and price data, yielding more explainable forecasts (Yu et al., 2023). Fine-tuning pre-trained language or image models — without modifying their core self-attention or feed-forward layers — has demonstrated effectiveness across diverse time-series tasks, with self-attention exhibiting behavior analogous to Principal Component Analysis (Zhou et al., 2023). Finally, the Instruct-FinGPT model illustrates the versatility of instruction-tuned LLMs for financial sentiment analysis (Zhang et al., 2023). Zhang et al. (2025a) systematically survey the field from statistical methods to LLMs, highlighting LLMs' superior zero-shot summarization. Pan et al. (2024) introduce S²IP-LLM, which aligns pretrained semantic anchors with decomposed time-series embeddings for prompt-based forecasting, demonstrating the necessity of semantic-informed prompts. Liu et al. (2025) propose an LLM-based multivariate forecaster that uses cross-modality alignment to extract robust embeddings from text and numerical data, outperforming traditional baselines. Finally, Jia et al. (2024) introduce GPT4MTS, a prompt-based LLM framework on a GDELT-derived dataset that effectively fuses text and numerical series, achieving significant accuracy gains in multimodal time-series prediction.

Although these methods have advanced their respective domains, no existing framework concurrently achieves all of the following: (1) hierarchical semantic abstraction and fine-grained fusion of textual and price time series; (2) multi-scale temporal modeling under tight computational budgets; and (3) resilience to extreme market shocks. To fill this gap, we introduce a novel multimodal forecasting architecture that exploits the semantic acuity of pretrained large language models through a two-stage hierarchical summarization pipeline for information extraction. Our design features a patch-based attention encoder that jointly represents interleaved text and price sequences, followed by a co-attention fusion module to enable deep cross-modal interactions. Experimental evaluation on benchmark datasets indicates that our framework outperforms state-of-the-art methods in both forecasting accuracy and robustness (See Section 4.5.).

---

## 3. Proposed method

### 3.1. Overview of our proposed model

We propose a four-stage framework that (i) hierarchically abstracts textual content, (ii) encodes fine-grained temporal patterns, and (iii) deeply fuses semantic and price dynamics for robust stock trend forecasting. For example, in the case of Apple Inc. (AAPL), one trading day's news and corresponding price series are processed by the multimodal framework illustrated in Fig. 1, which consists of four sequential modules:

**(1) Hierarchical Summary.** Hierarchical Summarization distills salient information and suppresses micro-level noise by first segmenting all textual sources (news articles, reports, social-media commentary) into one-hour intervals. Each hourly batch is compressed by a pre-trained LLM into a concise summary. These summaries are then concatenated in chronological order and reprocessed by the same LLM to generate a unified daily synopsis. Finally, a dense embedding vector is extracted from this daily synopsis, yielding a compact representation of the day's textual content.

**(2) Embedding and Alignment Module.** This stage bridges textual and numerical modalities through a cross-attention mechanism. Text embeddings and normalized price-series features serve as queries, keys, and values in a cross-attention block, integrating semantic and financial signals. The aligned sequences are subsequently partitioned into overlapping temporal patches. For each modality, a unidirectional Transformer encoder — with multi-head self-attention — processes these patches to capture local temporal dependencies. A trainable linear projection then maps both text and price patch embeddings into a shared latent space, and modality-specific self-attention layers further refine their internal structure.

**(3) Co-Attention Fusion Module.** Here, bidirectional cross-modal dependencies are captured via a symmetric dual-stream attention architecture. In one stream, text embeddings query price keys/values; in the other, price embeddings query text keys/values. Each stream produces an attention-weighted representation of cross-modal interactions. The outputs are concatenated along the feature axis and passed through a linear projection to restore original embedding dimensions. A residual connection merges this projection with the initial embeddings, and layer normalization stabilizes the fused tensor, which now unifies modality-specific and cross-modal context for downstream classification.

**(4) Classification Module.** The fused feature tensor is flattened into a feature vector, which is passed through a fully connected layer and a sigmoid activation to produce a probability score for upward price movement. Technical details for each module are given in Sections 3.2–3.5.

Our framework uses a fixed, task-specific LLM for two-stage summarization — first hourly, then daily — ensuring stable semantic abstraction without fine-tuning overhead. Only the encoding, fusion, and classification layers are trained end to end, enabling the temporal encoder to iteratively refine its features based on fusion feedback. By decoupling summarization from model training and focusing optimization on cross-modal integration, our design achieves efficient information extraction and comprehensive temporal insight, leading to superior predictive accuracy and robustness under volatile market conditions (see Section 4).

### 3.2. Hierarchical summary

**Algorithm 1** Hierarchical summarization.

```
Require: Raw documents {D_t}_{t=1}^{T}, hourly prompt Prompt_hour, daily prompt Prompt_day
Ensure: Daily summary embedding x_e

1:  // Hourly-level summarization
2:  for t = 1 to T do
3:      S_t ← LLM(Prompt_hour, D_t)
4:  end for
5:  // Concatenate hourly summaries
6:  S_concat ← concat(S_1, …, S_T)
7:  // Daily-level summarization
8:  S_day ← LLM(Prompt_day, S_concat)
9:  // Embedding extraction
10: x_e ← Embed(S_day)
11: return x_e
```

![Figure 1. Overview of the proposed model framework. Given a stock-related text sequence, the model summarizes and encodes it with the LLM, aligns the resulting embeddings with the normalized price series in the Embedding and Alignment Module, fuses them via co-attention, and projects the fused features to produce predictions.](images/fig1_framework_overview.png)
*Fig. 1. Overview of the proposed model framework. Given a stock-related text sequence, we summarize and encode it with the LLM, align the resulting embeddings with the normalized price series in the Embedding and Alignment Module, fuse them via co-attention, and project the fused features to produce predictions.*

![Figure 2. Trend Analysis and Driving Factors of AAPL Stock Prices (January 1 to February 12). Red indicates stock price increase; green indicates stock price decrease. Dashed boxes highlight days of pronounced price movements, corresponding to key events such as quarterly earnings releases, major product announcements, and significant macroeconomic data releases.](images/fig2_aapl_trend_analysis.png)
*Fig. 2. Trend Analysis and Driving Factors of AAPL Stock Prices (January 1, to February 12). Red indicates stock price increase; green indicates stock price decrease. Dashed boxes highlight days of pronounced price movements, corresponding to key events such as quarterly earnings releases, major product announcements, and significant macroeconomic data releases, which collectively shaped investor sentiment and drove heightened volatility.*

Our proposed hierarchical summarization module is grounded in Information Fusion Theory (Xu & Xu, 2017) and the principles of Multiresolution Signal Processing (Mallat, 1989), employing a multi-stage summarization strategy to alleviate information redundancy and sequence-length constraints inherent in large-scale heterogeneous text corpora. Specifically, guided by the Information Bottleneck Principle, we implement an *hourly-level summarization* routine that identifies and extracts the most information-rich sentences within each one-hour interval, thereby maximizing retention of critical event semantics while suppressing irrelevant noise. In a subsequent *daily-level summarization* pass, these hourly summaries are further distilled into a coarse-grained daily synopsis, yielding a macroscopic overview that balances local detail with global structure — a hallmark of multiresolution analysis. Finally, to respect the input-length limits and computational budget of large language models, only the resulting summary embeddings (rather than full-text sequences) are forwarded to downstream modules, substantially reducing token processing overhead and ensuring seamless, scalable integration with subsequent encoders.

Fig. 1 illustrates the architecture and workflow of the hierarchical summarization module (Wang et al., 2024). Algorithm 1 presents the pseudocode for the Hierarchical Summarization module. To accommodate the input-length limits and computational constraints of large language models when processing extensive stock-related text, a hierarchical summarization strategy is employed. Stock-related documents often contain fine-grained details that exceed typical sequence-length thresholds. Accordingly, the raw text is divided into hourly segments, each covering a fixed interval of stock-specific content. This segmentation preserves essential contextual and temporal relationships while reducing computational overhead. We apply this approach to the AAPL stock data plotted in Fig. 2, "Trend Analysis and Driving Factors of AAPL Stock Prices (January–February 12)." By selecting several days with pronounced price swings, we demonstrate how our hierarchical summaries capture the interplay between market events — such as earnings releases, shifts in global sentiment, and regulatory news — and observed volatility. The hourly-level summaries retain intraday dynamics (e.g., immediate reactions to news), while the daily-level summaries highlight the overarching drivers behind each significant move. Embedding these summaries then enables our prediction model to leverage both fine-grained temporal patterns and aggregated daily context, yielding a richer understanding of AAPL's price behavior over the January–February 12 period.

The workflow comprises three stages:

1. **Hourly-Level Summarization.** Raw text is divided into fixed-interval segments (typically one hour each) so that each segment can be processed by a pre-trained LLM. The model independently condenses each segment, retaining its key details. Resulting hourly summaries are then concatenated on a daily basis, maintaining temporal coherence and markedly reducing overall sequence length.
2. **Daily-Level Summarization.** The concatenated hourly summaries undergo a second pass with the LLM to generate a daily summary. A tailored prompt directs the model to extract only the most salient information, yielding a concise representation that omits redundant or less critical content. This additional condensation further shortens the text for downstream tasks.
3. **Embedding Extraction.** Embedding vectors are derived from the daily summaries, producing compact representations that capture both semantic and contextual nuances. These embeddings serve as input features for subsequent trend-prediction models, reflecting the hourly temporal dependencies and the aggregated daily overview.

### 3.3. Embedding and alignment module

**Text Embedding and Dimension Alignment.** Raw textual inputs are processed by the hierarchical summarization module to produce embeddings X_e ∈ ℝ^(B×L×d), where B denotes batch size, L the sequence length after summarization, and d the embedding dimension. This step reduces the length of textual sequences while preserving their semantic content, thereby lowering memory footprint and computational cost. Hourly summaries are consolidated into daily aggregates to synchronize with the temporal granularity of price data. Guided by cross-modal representation learning (Peng & Qi, 2019) and subspace alignment theory (Fernando et al., 2013), we embed textual and numerical signals into a common latent space in which their geometric relationships can be directly compared. Subspace alignment holds that projecting heterogeneous features into a shared subspace minimizes distributional discrepancy and facilitates meaningful similarity measures. Accordingly, we apply a learnable linear mapping followed by dropout regularization to adjust the embedding dimension from d to the price-feature dimension d_p:

> **X_t = Dropout(Linear(X_e))**,  X_t ∈ ℝ^(B×L×d_p)   (1)

Here, Linear denotes a linear layer and Dropout helps prevent overfitting by randomly zeroing a fraction of units during training.

**Feature Alignment.** To efficiently integrate textual information with price features, a cross-attention mechanism is employed to align the text embeddings with the price features. Specifically, given text embeddings X_t and price features X_p ∈ ℝ^(B×L×d_p), the cross-attention mechanism facilitates the comparison and integration of information from these two distinct sources, producing fused features that capture both textual content and price movements.

The query Q, key K, and value V vectors are computed for both the text and price features. The price features X_p are projected into queries Q using the weight matrix W_q, while the text embeddings X_t are projected into keys K and values V using the weight matrices W_k and W_v, respectively. The dot product between the query and key is then computed to assess the relationship between the text and price, as described in Eq. (2), which produces an attention distribution that quantifies the influence of each text segment on price prediction.

Subsequently, the attention distribution is normalized using the softmax function, resulting in a weighted output feature X_tp ∈ ℝ^(B×L×d_p), as shown in Eq. (3), which integrates both price and text information. This enables the final feature to reflect both market price fluctuations and the latent information within the text. The cross-attention mechanism thus aligns the price and text embeddings and enhances their interaction through efficient computation, leading to improved feature representations for subsequent trend prediction tasks.

By employing this cross-attention (Vaswani et al., 2017) alignment approach, price and text data are integrated into a shared representation space, enhancing the model's ability to understand and predict market fluctuations, especially in modeling the relationship between stock market volatility and news events.

> **Q = X_p W_q,  K = X_t W_k,  V = X_t W_v**   (2)

with W_q, W_k, W_v ∈ ℝ^(d_p×d_k) and d_k = d_p. Attention weights are computed as:

> **X_tp = softmax(QKᵀ / √d_k) V**   (3)

**Patch Partitioning and Position Projection.** Long sequences in X_tp and X_p are segmented into N overlapping patches of length P with stride S:

> **N = ⌊(L − P)/S⌋ + 2,  X_tp^pa, X_p^pa ∈ ℝ^(B×d_p×N×P)**   (4)

This patching strategy is motivated by multiresolution signal-processing theory (Mallat, 1989) and by empirical results in computer vision and speech (Tang et al., 2022), both of which show that local context windows provide an effective inductive bias for modelling long sequences. Concretely, we partition both the text and price streams into length-P windows that overlap by a stride S < P. From an information-theoretic viewpoint, each window serves as a local receptive field, allowing the encoder to capture high-frequency nuances — such as minute-level sentiment shifts or micro-price jumps — without being overwhelmed by global context. The overlap mitigates the boundary problem: events that straddle two adjacent windows appear (in full) in at least one of them, thereby preserving continuity across segments.

Computationally, patching converts the quadratic self-attention cost of a length-L sequence, O(L²), into the sum of N = ⌊(L − P)/S⌋ + 2 local attentions with complexity O(NP²). When P ≪ L, this reduces both floating-point operations and memory use by roughly an order of magnitude, enabling mini-batch training on commodity GPUs even for intraday data sampled at high frequency. Because each patch is processed independently before the global fusion stage, all windowed projections and positional encodings run in parallel on modern accelerators. This embarrassingly parallel design scales linearly with sequence length once adequate processing threads are available, providing a practical path for handling multi-year historical archives and fine-grained intraday feeds without prohibitive latency.

Position embeddings and a learnable linear projection are then applied to both modalities in an identical manner to maintain consistency in feature transformation across both textual and price data. This ensures that both types of information are treated with equal importance in the subsequent stages of the model. The linear projection transforms the patch representations into a shared embedding space, while the position embeddings encode the relative position of each patch, maintaining the sequential order within the input data.

The mathematical representation of this operation is as follows:

> **X̃ = X^pa W_proj + W_pos,  W_proj ∈ ℝ^(P×D), W_pos ∈ ℝ^(B×d_p×N×D), X̃ ∈ ℝ^(B×d_p×N×D)**   (5)

Here, D denotes the embedding dimension, and the parameters W_proj and W_pos are learnable, optimizing the representation of both patches and positions to enhance the model's ability to integrate the two modalities. By employing this patch-based approach, a balance between computational efficiency and the preservation of key semantic and temporal dependencies in the input data is maintained.

**Self-Attention Enhancement.** Following the patch partitioning process, the reshaped patches are further refined using self-attention (Vaswani et al., 2017) to emphasize the most critical features. Self-attention is applied to capture dependencies within each patch, allowing the model to focus on the most relevant components of the data, regardless of their position in the sequence. This mechanism enables the model to dynamically adjust the weight given to different parts of the data, ensuring that essential semantic and temporal relationships are highlighted.

The self-attention operation begins by computing the query (Q), key (K), and value (V) matrices from the reshaped patches, represented by X̂. Here, the price-feature dimension d_p is flattened into the batch axis, resulting in a standard 3D input tensor (batch, sequence, channel) for attention modules without information loss. The query, key, and value matrices are computed as:

> **Q = X̂ W_q,  K = X̂ W_k,  V = X̂ W_v,  X̂ ∈ ℝ^(Bd_p×N×D)**   (6)

where W_q ∈ ℝ^(D×d_q), W_k ∈ ℝ^(D×d_k), and W_v ∈ ℝ^(D×d_v) are the learnable weight matrices for the query, key, and value projections, respectively, and Bd_p denotes the size of the first dimension of X̂.

The self-attention mechanism computes the dot product between the query and key matrices to determine the attention scores, which are then normalized using the softmax function. This process produces a refined feature representation, X^h, which highlights the critical features within the patches. The output is computed as:

> **X^h = softmax(QKᵀ / √d_k) V,  X^h ∈ ℝ^(Bd_p×N×d_v)**   (7)

Self-attention enables the model to capture long-range dependencies and interactions within each patch, including relationships between elements that are distant in the original sequence. This mechanism highlights salient features before they proceed to subsequent components, such as the Co-Attention Fusion module (Section 3.4). When combined with patch partitioning, self-attention facilitates the learning of both local and global dependencies, accommodating the complexity of textual and price data in trend-prediction tasks.

### 3.4. Co-attention fusion

Early-fusion concatenation ignores inter-modal structure (Xu et al., 2020; Xu & Cohen, 2018), while one-way cross-attention enforces a fixed hierarchy between modalities. Motivated by bidirectional vision-language architectures (Zhang et al., 2024b) and dual-encoder alignment theory (Bao et al., 2022), we introduce a bidirectional co-attention block. In this design, text attends to price so that news highlights market-relevant regions of the price trace, and price attends back to text so that price movements contextualise the interpretation of headlines. The two reciprocal streams produce complementary feature maps that are concatenated and linearly projected to yield a unified, context-aware embedding.

Compared with early fusion, the proposed module preserves modality-specific structure; compared with single-direction cross-attention, it captures feedback loops that are intrinsic to financial decision making.

Let X_t^h, X_p^h ∈ ℝ^(Bd_p×N×d_v) be the patch-level text and price tensors (N patches, hidden width d_v). Here, X_t^h and X_p^h denote the text and price embeddings, respectively, after refinement via the self-attention enhancement. We use h heads; per-head width is d_h = d_v/h.

**Stream 1: text → price.** In the first layer, text embeddings are utilized as the query, while price embeddings are used as both the key and value. This design ensures that the text data, as the primary source of semantic information, attends to the price data, which contains temporal and market-specific features.

For head i:

> **Q^(i) = X_t^h W_q1^(i),  K^(i) = X_p^h W_k1^(i),  V^(i) = X_p^h W_v1^(i)**   (8)

where W_q1^(i), W_k1^(i), W_v1^(i) ∈ ℝ^(d_v×d_h) are learnable projection matrices. The attention output is then computed as:

> **head_1^(i) = softmax(Q^(i) K^(i)ᵀ / √d_h) V^(i)**
> **O_1 = Concat(head_1^(1), …, head_1^(h)) W_o,  W_o ∈ ℝ^(d_v×d_v)**   (9)

This attention mechanism allows the model to selectively focus on important price-related features while processing the text data, which enhances the model's ability to interpret the interaction between the two types of data.

**Stream 2: price → text.** In the second co-attention layer, the roles of the query, key, and value are reversed. In this case, price embeddings serve as the query, while text embeddings act as both the key and value. This approach enables the price data to attend to the text data, allowing the model to capture important semantic features that influence price dynamics. The attention operation in this layer is defined as:

> **Q'^(i) = X_p^h W_q2^(i),  K'^(i) = X_t^h W_k2^(i),  V'^(i) = X_t^h W_v2^(i)**   (10)

where W_q2^(i), W_k2^(i), W_v2^(i) ∈ ℝ^(d_v×d_h). With the corresponding attention output given by:

> **head_2^(i) = softmax(Q'^(i) K'^(i)ᵀ / √d_h) V'^(i)**
> **O_2 = Concat(head_2^(1), …, head_2^(h)) W_o',  W_o' ∈ ℝ^(d_v×d_v)**   (11)

This layer allows the model to emphasize relevant textual information that can influence price movements, further refining the interaction between the two modalities.

The outputs from these two co-attention layers are then concatenated along the last dimension to combine the refined feature representations from both layers. A linear transformation is applied to the concatenated output to generate the final joint representation:

> **Ô = Linear(Concat(O_1, O_2)),  Ô ∈ ℝ^(Bd_p×N×d_v)**   (12)

This step ensures that the model incorporates information from both directions of the co-attention mechanism, optimizing the fusion of textual and price features.

To improve training stability and prevent overfitting, a residual connection is added to the output of the linear transformation, followed by layer normalization. This process ensures that the learned representations retain both the original features and the enhanced co-attention information. The residual connection and layer normalization are computed as:

> **Z_1 = LayerNorm(Ô + X_p^h),  Z_2 = FFN(Z_1),  Z = LayerNorm(Z_1 + Z_2)**   (13)

where Z ∈ ℝ^(Bd_p×N×d_v) is the final joint feature representation.

A residual connection is applied to the output of the linear projection, and the combined tensor is then normalized via layer normalization to promote stable convergence and mitigate overfitting. This arrangement preserves the original feature signals alongside the enriched co-attention representations.

**Computational profile and generality.** With h heads of width d_h = d_v/h, each co-attention stream executes h(N²d_h) = O(N²d_v) operations per batch — quadratic only in the patch count N ≪ L. Because all heads and both streams run concurrently, the fusion block is embarrassingly parallel on modern GPUs and remains tractable even for minute-level intraday sequences.

The architecture itself is modality-agnostic: the price channel can be replaced by any numeric time series — e.g., vital signs in electronic health records, traffic flow paired with alert messages, or energy demand with weather bulletins — without altering model structure. Hence the bidirectional co-attention template offers a reusable blueprint for cross-modal temporal modelling.

In sum, the proposed mechanism preserves modality-specific structure while capturing rich inter-modal dependencies, producing a semantically informed joint representation that benefits downstream tasks such as stock-trend classification.

### 3.5. Classification module

The final joint feature tensor Z is first reshaped into a two-dimensional matrix by concatenating the modality and patch dimensions. Specifically, the flattening operation yields:

> **z_f = flatten(Z),  z_f ∈ ℝ^(Bd_p×Nd_v)**   (14)

This vector is passed through a fully connected layer followed by a sigmoid activation to yield the predicted trend probability:

> **ŷ = σ(z_f W_c + b),  ŷ ∈ ℝ^(Bd_p×l)**   (15)

where W_c ∈ ℝ^(Nd_v×l), b ∈ ℝ^l, and σ(x) = 1/(1 + e^(−x)).

Model parameters {W_c, b} are optimized by minimizing the binary cross-entropy loss over the training set:

> **BCE = −[y log ŷ + (1 − y) log(1 − ŷ)]**   (16)

where y ∈ {0, 1}^((B d_p)×l) is the ground-truth label. This configuration ensures the output ŷ represents the probability of an upward price movement.

---

## 4. Experiments

To evaluate the effectiveness and robustness of the proposed multimodal framework, a comprehensive experimental protocol was established. Section 4.1 describes the CMIN-US and CMIN-CN benchmarks. Section 4.2 presents the problem formulation and experimental setup. Section 4.3 defines the evaluation metrics. Section 4.4 compares performance against six state-of-the-art baselines. Section 4.5 reports the main results and accompanying analysis. Section 4.6 details the ablation study quantifying the contribution of each component. Section 4.7 provides a case study of two extreme AAPL price swings, using PCA-based quadrant analysis to illustrate how the hierarchical-summary module sharpens model interpretability and improves prediction accuracy. Finally, Section 4.8 examines the sensitivity of key hyperparameters.

### 4.1. Datasets

We assess our framework on two publicly available cross-market equity datasets, CMIN-US and CMIN-CN (Luo et al., 2023a), which provide synchronized financial text and price series. The datasets cover the period from January 1, 2018 to December 31, 2021 and are available under an open access license at https://github.com/BigRoddy/CMIN-Dataset.

- **CMIN-US.** This dataset comprises daily closing prices and corresponding news headlines for the 110 largest U.S. stocks by market capitalization. Price data were obtained from Yahoo Finance, and textual data were retrieved via the Yahoo Finance news API.
- **CMIN-CN.** This dataset includes daily closing prices and financial news for all 300 constituents of the CSI 300 index. Historical price series were sourced from Wind, while news articles were collected from the Wind Information platform.

**Table 1. Dataset descriptions.**

| Dataset | Description | Data Type | Data Range |
|---|---|---|---|
| CMIN-US | US: 110 Companies | Time Series & Text | 2018-01-01 to 2021-12-31 |
| CMIN-CN | China: 300 Companies | Time Series & Text | 2018-01-01 to 2021-12-31 |

**Table 2. Hyperparameter settings.**

| Type | Parameter | Value |
|---|---|---|
| Learning Parameters | Learning Rate | 0.0001 |
| Learning Parameters | Learning Rate Decay Rate | 0.0001 |
| Learning Parameters | Learning Rate Decay Epoch | 5 |
| Learning Parameters | Epoch | 20 |
| Learning Parameters | Batch Size | 16 |
| Learning Parameters | Patience | 5 |
| Structure Parameters | Sequence Length L | 30 |
| Structure Parameters | Patch Length | 10 |
| Structure Parameters | Stride | 5 |
| Structure Parameters | Number of Classes | 1 |
| Structure Parameters | Number of Layers | 4 |
| Structure Parameters | d_model | 128 |
| Structure Parameters | d_ff | 256 |
| Structure Parameters | Dropout | 0.2 |
| Structure Parameters | Activation | GELU |

Table 1 reports summary statistics for each dataset, including the number of stocks, total trading days, average headline length, and data-preprocessing procedures (e.g., missing-value imputation, headline deduplication, tokenization).

### 4.2. Problem formulation and experimental setup

We define the prediction task and label construction before detailing the model configuration. We cast stock movement forecasting as a binary next-day trend prediction problem: given price sequences and contemporaneous text up to day τ, predict whether the closing price on day τ+1 increases relative to day τ. Let x_τ denote the closing price on day τ. The target variable y_(τ+1) is

> **y_(τ+1) = 1, if x_(τ+1) > x_τ (upward); 0, if x_(τ+1) ≤ x_τ (downward/no change).**   (17)

This binary formulation aligns with trading decisions and is widely adopted in multimodal stock prediction (Li et al., 2024b; Luo et al., 2023b). For each prediction at day τ, the model consumes two time-aligned modalities: (i) a price sequence — an L=30 trading-day window of closing prices X_p^(τ) = [x_(τ−L+1), …, x_τ] ∈ ℝ^L, standardized via a training-set z-score transform x̃_i = (x_i − μ_train)/σ_train, where μ_train and σ_train are computed on the training data only to avoid leakage; and (ii) a textual sequence — all financial news within the same L-day window, processed by the hierarchical summarization module (Section 3.2) to yield a daily embedding X_e^(τ) ∈ ℝ^d. Unlike approaches that impose magnitude thresholds to filter small moves (Han et al., 2023b; Ma et al., 2023b), we apply no threshold: any x_(τ+1) > x_τ is labeled upward (y=1); otherwise y=0. This strict policy enables learning directional signals at all scales, including subtle momentum that may precede larger moves. Empirically, daily movement distributions are well behaved: in CMIN-US, 52.3% of days are positive with a mean absolute daily change of 1.87%; in CMIN-CN, 48.6% are positive with 2.14% mean absolute change, indicating naturally reasonable class proportions without thresholding.

To avoid look-ahead bias, we chronologically split the full sample (2018-01-01 to 2021-12-31; ~1008 trading days) into training (2018-01-01 to 2020-06-30; ~630 days, ~62.5%), validation (2020-07-01 to 2020-12-31; ~126 days, ~12.5%), and test (2021-01-01 to 2021-12-31; ~252 days, ~25%). For each stock s and day τ with τ ≥ L and τ < T−1 (where T is the last trading day of the split), we construct a sample using the L-day window and the next-day label, yielding approximately 69,300 training samples for CMIN-US (110 stocks × 630 days) and 189,000 for CMIN-CN (300 stocks × 630 days), with proportional validation and test sets. Labels exhibit mild, realistic imbalance, so we report Matthews Correlation Coefficient (MCC) alongside accuracy (Section 4.3) to fairly assess both classes without resampling or class weights (Chicco & Jurman, 2020). All metrics are computed on the held-out 2021 test set using models trained on 2018–H1 2020 and validated on H2 2020, ensuring genuine out-of-sample generalization.

For model configuration, we adopt the pre-trained mT5_multilingual_XLSum model (Hasan et al., 2021), a multilingual summarization model built on the T5 (Text-to-Text Transfer Transformer) architecture and trained on the XL-Sum corpus. Its transformer framework provides rich representations from complex text and is well suited to multilingual summarization. Our architecture comprises three modules — Alignment & Embedding, Co-Attention Fusion, and Classification — with sequence length L=30, a patch size of 10 and stride of 5 for temporal chunking, hidden dimension d_model=128, and feed-forward dimension d_ff=256. We use GELU activations and a dropout rate of 0.2 throughout. The output layer produces a single probability for binary trend classification. Training uses an initial learning rate of 1×10⁻⁴ with a decay of 1×10⁻⁴ applied every five epochs; unless otherwise stated, other training details follow standard practice, with further hyperparameter details shown in Table 2.

**Preprocessing and data integrity.** To ensure data quality and model reliability, we implement a multi-stage preprocessing pipeline tailored to common issues in financial time series. We address missing trading days caused by holidays, trading halts, or collection errors by forward-filling the last observed close when a gap is ≤ 3 consecutive days; if, over 2018–2021, the number of missing days for a stock exceeds 5% of total trading days (indicating severe data quality problems), the stock is excluded from analysis. This procedure removes 2 stocks from CMIN-US and 4 from CMIN-CN, leaving 108 and 296 stocks, respectively, for evaluation. To mitigate extreme price swings that can distort normalization statistics, we apply a 3-sigma rule independently to each stock s: using training-period daily log returns, we compute μ_s and σ_s and truncate any return outside μ_s ± 3σ_s to the boundary. Contemporaneous textual activity is cross-checked to preserve informative events associated with genuine large moves. Across both datasets, fewer than 0.8% of daily returns are truncated. To reduce textual redundancy from repeated wire stories, we compute pairwise Levenshtein distances among all articles for the same stock on the same trading day and keep only the earliest timestamped version when the character-level overlap exceeds 90%; this reduces article counts by approximately 18% for CMIN-US and 22% for CMIN-CN and focuses modeling on unique information. We then normalize raw news text by removing non-alphanumeric characters, collapsing repeated whitespace, filtering articles shorter than 50 characters, and lowercasing all text; the cleaned corpus is tokenized with the pre-trained mT5 tokenizer under its 512-token input limit. All preprocessing hyperparameters are fixed on the training set (2018–H1 2020) and applied unchanged to validation and test sets to prevent information leakage.

To avoid look-ahead bias, we use a chronological split with zero temporal overlap: training, validation, and test. Model selection, hyperparameter tuning, and early stopping are performed exclusively on the validation set; the test set is evaluated once after model selection. Upward-label shares are naturally close to balanced and reflect genuine market dynamics rather than labeling artifacts.

### 4.3. Evaluation metrics

To rigorously assess predictive performance and mitigate class-imbalance effects, we employ two complementary metrics: classification accuracy (ACC) and the Matthews correlation coefficient (MCC). ACC measures overall correctness, while MCC provides a balanced evaluation even when class frequencies differ.

> **ACC = (TP + TN) / (TP + TN + FP + FN)**   (18)

> **MCC = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]**   (19)

where:
- TP (True Positive): Correct predictions of an upward trend,
- TN (True Negative): Correct predictions of a downward trend,
- FP (False Positive): Type I errors in trend prediction,
- FN (False Negative): Type II errors in trend prediction.

### 4.4. Baselines

We benchmark our proposed model against six state-of-the-art stock trend prediction methods:

- **ALSTM** (Qin et al., 2017): a hierarchical attention-based recurrent network that employs dual-level attention to selectively weight temporal features across all historical time steps.
- **StockNet** (Xu & Cohen, 2018): a variational recurrent framework integrating recursive continuous latent variables, which leverages variational inference to address the posterior estimation challenge in time-series modeling.
- **Adv-LSTM** (Feng et al., 2019): an adversarially trained variant of ALSTM designed to improve model robustness and generalization under distributional shifts.
- **DTML** (Yoo et al., 2021): an attention-driven architecture that explicitly incorporates inter-stock correlation matrices to refine feature representations and enhance directional prediction accuracy.
- **CMIN** (Luo et al., 2023a): an end-to-end multimodal network that fuses financial text embeddings with causally augmented stock correlation graphs, enabling joint modeling of textual and numerical modalities.
- **LLMFactor** (Wang et al., 2024b): a prompt-based method leveraging large language models to generate sequence-guided prompts, which identify and weight key textual factors influencing stock movements.
- **CausalStock** (Li et al., 2024c): a news-driven framework that infers lag-dependent causal relations among equities and employs an LLM-based denoiser to enhance multi-stock movement prediction.

### 4.5. Performance analysis

Table 3 presents a comprehensive performance comparison across CMIN-US and CMIN-CN benchmarks, revealing how our hierarchical-patch-co-attention framework addresses fundamental limitations in existing multimodal stock prediction methods through task-aligned semantic distillation, efficient temporal modeling, and explicit cross-modal interaction. Existing approaches demonstrate significant performance gaps. Unimodal models relying solely on historical prices achieve approximately 51–52% accuracy — barely above random guessing — indicating that single-modal signals lack sufficient discriminative power for effective trend forecasting. Introducing textual information yields improvements, yet the magnitude varies dramatically with fusion strategy. StockNet and DTML leverage textual data alongside inter-stock correlations to reach 52–54% accuracy, while CMIN exhibits limited gains despite causally structured embeddings, suggesting ineffective fusion can introduce redundancies that degrade performance. Most notably, LLMFactor represents the strongest baseline at 64.46% accuracy but still fails to sufficiently align textual and price representations at fine-grained levels.

**Table 3. Performance comparison on CMIN-US and CMIN-CN (mean ± std over 10 runs).**

| Model | CMIN-US ACC (%) | CMIN-US MCC | CMIN-CN ACC (%) | CMIN-CN MCC |
|---|---|---|---|---|
| ALSTM | 51.64 ± 0.15 | 0.006 ± 0.003 | 53.35 ± 0.12 | 0.023 ± 0.004 |
| StockNet | 52.46 ± 0.18 | 0.022 ± 0.005 | 54.53 ± 0.14 | 0.045 ± 0.006 |
| Adv-LSTM | 51.73 ± 0.16 | 0.012 ± 0.004 | 53.49 ± 0.13 | 0.025 ± 0.005 |
| DTML | 52.06 ± 0.17 | 0.031 ± 0.005 | 54.42 ± 0.15 | 0.083 ± 0.007 |
| CMIN | 53.43 ± 0.20 | 0.046 ± 0.006 | 55.28 ± 0.16 | 0.111 ± 0.008 |
| LLMFactor | 64.46 ± 0.22 | 0.267 ± 0.010 | 57.96 ± 0.19 | 0.194 ± 0.009 |
| CausalStock | 54.60 ± 0.19 | 0.048 ± 0.007 | 56.25 ± 0.15 | 0.142 ± 0.008 |
| **Proposed** | **67.01 ± 0.21** | **0.346 ± 0.007** | **70.24 ± 0.21** | **0.293 ± 0.010** |

Our method achieves 67.01% accuracy and 0.346 MCC on CMIN-US, alongside 70.24% accuracy and 0.293 on CMIN-CN, outperforming LLMFactor by +2.55 percentage points in accuracy and +0.079 in MCC. These gains stem from three synergistic architectural innovations. The two-stage hierarchical summarization pipeline compresses raw financial news at hourly intervals to preserve intraday event structure — capturing immediate market reactions to earnings announcements, regulatory filings, or macroeconomic releases — then distills these summaries into daily synopses that filter micro-level noise while respecting LLM input-length constraints. This multi-resolution extraction keeps the model focused on market-moving signals rather than transient fluctuations or redundant repetitions. The patch-based temporal encoder partitions sequences into overlapping windows, reducing attention complexity from O(L²) to O(NP²). Because the number of patches N scales roughly with L/S and is therefore much smaller than L under typical settings, this yields a several-fold speedup while preserving both local and global temporal structure. Unlike full-sequence attention struggling to capture intraday spikes and multi-day momentum simultaneously, overlapping patches ensure critical events appear completely within at least one window, mitigating boundary artifacts and distinguishing genuine directional shifts from noise.

The bidirectional text–price co-attention mechanism explicitly models feedback loops intrinsic to financial decision-making: textual embeddings query price features to identify relevant patterns given news sentiment, while price embeddings query text to contextualize movements with semantic signals. This symmetric architecture captures mutual reinforcement — positive earnings news amplifying rising price signals, sustained increases lending credibility to optimistic narratives — substantially outperforming unidirectional cross-attention and naive concatenation.

Performance enhancements are statistically significant under paired t-test at p<0.01, confirming reliability rather than artifacts of random initialization. Consistent gains across distinct market contexts under different regulatory regimes — underscore robustness and generalization. The framework learns representations of how textual semantics and price dynamics jointly encode signals rather than overfitting to single-market idiosyncrasies.

### 4.6. Ablation study

To disentangle the contribution of each design choice, we conduct three controlled experiments: (i) modality contribution; (ii) text summarisation strategy; (iii) fusion mechanism. Unless otherwise specified, results are reported as the mean and standard deviation over 10 independent runs, and statistical significance is assessed using a paired t-test.

**Modality contribution and feature ablation.** Table 4 systematically evaluates input modalities and feature engineering strategies. Unimodal baselines show limited performance: text-only achieves 56.65% ACC on CMIN-US, while Close-only attains 58.12%. To address close-centric concerns, we progressively augment price inputs with complementary features.

Expanding from Close to six-dimensional OHLCV — comprising Open, High, Low, Close, and Volume — improves accuracy to 59.99% on CMIN-US, representing a gain of 1.87 percentage points and demonstrating that intraday variability and trading activity provide valuable signals. Incorporating 11 technical indicators, denoted as TA, further boosts performance by capturing momentum, volatility clustering, and mean-reversion dynamics. Combined with the 6-dimensional OHLCV features, this yields a 17-dimensional price representation spanning momentum (returns, RSI, MACD), volatility (Bollinger Bands, ATR), trend (moving averages), and volume (OBV) dimensions.

Integrating OHLCV with hierarchically summarized news via co-attention yields dramatic gains: accuracy jumps to 65.34% on CMIN-US and 69.97% on CMIN-CN, with MCC increasing to 0.342 — far exceeding technical indicator improvements alone. This confirms that textual semantics capture market-moving events orthogonal to quantitative patterns. The full model combining Price, TA, and Text achieves 67.01% ACC and 0.362 MCC on CMIN-US, alongside 70.24% and 0.308 on CMIN-CN, demonstrating that co-attention successfully integrates technical indicators with textual semantics for superior forecasting.

**Table 4. Feature ablation (mean ± std over 10 runs).**

| Experiment | Price Features | CMIN-US ACC (%) | CMIN-US MCC | CMIN-CN ACC (%) | CMIN-CN MCC |
|---|---|---|---|---|---|
| Text only | – | 56.65 ± 0.30 | 0.125 ± 0.008 | 59.10 ± 0.25 | 0.149 ± 0.010 |
| Close Only | 1 | 58.12 ± 0.11 | 0.131 ± 0.006 | 60.21 ± 0.11 | 0.151 ± 0.005 |
| Price: Close+OHLCV | 6 | 59.99 ± 0.22 | 0.169 ± 0.005 | 61.27 ± 0.20 | 0.171 ± 0.011 |
| Price+Tech Ind. | 17 | 60.85 ± 0.20 | 0.192 ± 0.007 | 61.85 ± 0.18 | 0.201 ± 0.009 |
| Price+Text | 6 | 65.34 ± 0.25 | 0.342 ± 0.011 | 69.97 ± 0.21 | 0.293 ± 0.010 |
| **Price+TA+Text** | **17** | **67.01 ± 0.21** | **0.362 ± 0.009** | **70.24 ± 0.19** | **0.308 ± 0.012** |

**Text summarization strategy.** As reported in Table 5, replacing direct embedding with single-stage summarization yields modest improvements; for example, accuracy increases from 62.00% to 63.42% on CMIN-US. However, our hierarchical summarization approach delivers the largest uplift, achieving 67.01% ACC and 0.346 MCC on CMIN-US — representing an increase of 3.34 percentage points in ACC and 0.022 in MCC over direct embedding — and 70.24% ACC and 0.293 MCC on CMIN-CN. These results confirm that hierarchical summarization condenses salient information while respecting LLM input constraints.

**Table 5. Effects of summary strategy (mean ± std over 10 runs).**

| Strategy | CMIN-US ACC (%) | CMIN-US MCC | CMIN-CN ACC (%) | CMIN-CN MCC |
|---|---|---|---|---|
| Direct embedding | 62.00 ± 0.28 | 0.320 ± 0.009 | 64.95 ± 0.26 | 0.248 ± 0.008 |
| Single-stage summary | 63.42 ± 0.27 | 0.329 ± 0.010 | 66.31 ± 0.24 | 0.258 ± 0.009 |
| **Hierarchical summary** | **67.01 ± 0.21** | **0.346 ± 0.007** | **70.24 ± 0.19** | **0.308 ± 0.012** |

**Fusion mechanism.** Table 6 examines three fusion schemes. A simple concatenate-plus-linear layer performs poorly, reaching only 59.42% ACC and 0.115 MCC on CMIN-US, indicating that naive feature joining fails to capture cross-modal dependencies. Introducing cross-attention dramatically raises accuracy to 64.47% and MCC to 0.312. Finally, our text-price co-attention delivers the best results at 67.01% ACC and 0.346 MCC on CMIN-US, further improving trend prediction by explicitly modeling fine-grained interactions between textual and price embeddings.

**Table 6. Comparison of fusion mechanisms (mean ± std over 10 runs).**

| Fusion Method | CMIN-US ACC (%) | CMIN-US MCC | CMIN-CN ACC (%) | CMIN-CN MCC |
|---|---|---|---|---|
| Concatenate + linear | 59.42 ± 0.20 | 0.115 ± 0.007 | 59.97 ± 0.18 | 0.167 ± 0.008 |
| Cross-attention | 64.47 ± 0.23 | 0.312 ± 0.010 | 66.65 ± 0.22 | 0.238 ± 0.009 |
| **Text-price co-attention** | **67.01 ± 0.21** | **0.346 ± 0.007** | **70.24 ± 0.19** | **0.308 ± 0.012** |

**Takeaways.** Across datasets, the incremental gains rank as follows: co-attention provides the largest improvement, followed by hierarchical summary, then modality fusion. These findings empirically underpin the design of our pipeline and highlight that deep cross-modal interaction — rather than merely adding more data — drives the majority of predictive improvement.

#### 4.6.1. Comparative LLM performance

To assess model stability, deployability, and the impact of LLM choice on forecasting performance, we conduct three complementary evaluations. First, we examine risk-adjusted returns and portfolio-level metrics via simulated trading. Second, we measure inference latency and computational scalability for real-time deployment. Third, we perform a systematic comparison of LLMs spanning general-purpose, domain-adapted, and summarization-pretrained architectures. Table 7 reports means and standard deviations over 10 independent runs with different random seeds to ensure statistical robustness. All models are trained on identical data splits and evaluated under a unified backtesting protocol with initial capital $100,000, round-trip transaction cost 0.1%, entry confidence threshold 0.6, and minimum holding period of 3 days.

**Risk-adjusted performance and economic significance.** Classification accuracy is informative, but practical trading systems require robust risk-adjusted returns under volatility, tail risk, and drawdown constraints. We therefore implement a simulated trading engine that goes long when the predicted upward probability exceeds 0.6 and stays in cash otherwise, with 0.1% transaction costs, a 3-day minimum holding period, and confidence-proportional position sizing. As shown in Table 7, the mT5-XLSum–based hierarchical summarization framework achieves the highest Sharpe ratio and superior annualized returns relative to BERT-base (Devlin et al., 2019), FinBERT (Araci, 2019), and XLM-RoBERTa (Liu et al., 2019). The gains arise from preserving event structure while filtering micro-level noise through two-stage hierarchical summarization, capturing both local dynamics and multi-day trends via patch-based temporal encoding, and explicitly modeling text–price interactions with bidirectional co-attention to surface high-conviction, asymmetric opportunities. Annualized returns corroborate the economic value, while downside risk is better controlled, yielding a higher Calmar ratio. The win rate reaches 56.7% ± 1.2% after costs — above the 50% positive-expectancy threshold — and, combined with confidence-weighted sizing, explains the Sharpe improvements.

**Inference efficiency, scalability, and real-time feasibility.** Real-time trading demands millisecond-level latency. Averaged over 1,000 forward passes on a single NVIDIA RTX 5080, mT5-XLSum attains 15.79 ± 0.48 ms per batch — about 1.97 ms per sample — comfortably within the 10–100 ms budget typical of intraday strategies. Despite a larger parameter count, latency remains competitive because patch-based encoding reduces attention complexity, and co-attention operates only on patch-level embeddings.

**Effect of LLM architecture on hierarchical summarization.** The pretrained language model used for hierarchical summarization materially affects semantic extraction and efficiency. Results in Table 7 show that mT5-XLSum consistently leads on both classification and portfolio metrics, indicating that task-aligned summarization pretraining is more consequential than domain vocabulary or multilingual coverage alone. Encoder–decoder denoising objectives encourage distillation of salient propositions and suppression of redundancy — well aligned with our hourly-to-daily summarization — whereas masked-LM objectives (e.g., BERT/FinBERT) emphasize local token reconstruction and provide smaller gains once texts have been condensed. XLM-RoBERTa offers the lowest latency but the weakest risk-adjusted returns; its multilingual pretraining yields discriminative features for classification yet does not consistently capture trading-salient sentiment and event salience after costs and holding constraints.

Taken together, Table 7 demonstrates that our hierarchical-summarization + patch-based encoding + co-attention framework improves risk-adjusted returns without introducing an online bottleneck: Sharpe increases materially, annual returns rise with shallower drawdowns, latency remains within real-time budgets, and summarization-pretrained mT5-XLSum proves most effective among tested LLMs.

**Table 7. LLM comparison for hierarchical summarization (mean ± std over 10 runs). Risk metrics from simulated trading on CMIN-US. Inference time per batch on NVIDIA RTX 5080.**

| LLM | Acc (%) | MCC | Sharpe | Ann. Ret. | Max DD | Win | Time (ms) |
|---|---|---|---|---|---|---|---|
| **mT5-XLSum** | **67.01 ± 0.21** | **0.346 ± 0.007** | **0.652 ± 0.028** | **0.245 ± 0.015** | **−0.182 ± 0.016** | **0.567 ± 0.012** | 15.79 ± 0.48 |
| BERT-base | 66.71 ± 0.20 | 0.310 ± 0.007 | 0.418 ± 0.021 | 0.156 ± 0.012 | −0.235 ± 0.022 | 0.534 ± 0.011 | 15.39 ± 0.47 |
| FinBERT | 66.95 ± 0.22 | 0.323 ± 0.004 | 0.574 ± 0.024 | 0.214 ± 0.014 | −0.198 ± 0.018 | 0.558 ± 0.011 | 15.76 ± 0.46 |
| XLM-R | 66.21 ± 0.25 | 0.335 ± 0.005 | 0.386 ± 0.020 | 0.142 ± 0.011 | −0.251 ± 0.024 | 0.529 ± 0.010 | **14.76 ± 0.45** |

### 4.7. Case study on extreme price movements

To highlight both the interpretability and effectiveness of our hierarchical-summary module, we analyse two extreme AAPL price swings in 2018 drawn from the CMIN-US test set. By contrasting hierarchical summaries with raw embeddings and single-stage summaries, we show how the hierarchical design steers the model toward the most informative cues and, in turn, improves predictive performance.

**Case A – Earnings beat (+7.3%) on 1 Nov. 2018.** On 1 November 2018, AAPL's quarterly earnings dramatically beat analyst estimates, pushing the stock up by 7.3%. When the model relies only on raw embeddings or a single-stage summary, it fails to capture the lasting impact of the beat: attention diffuses over generic terms and under-weights key phrases such as margin expansion, which attenuates the signal. With the hierarchical summary, hourly snippets are first extracted and then aggregated across the day, allowing the model to focus on core passages like EPS upgrade and raised guidance. In the PCA scatter plot (Fig. 3, upper-right quadrant), this focus appears as a pronounced "in-depth + bullish" cluster. Consequently, the one-day prediction accuracy rises by about 3.3 percentage points, and MCC improves by roughly 0.02, matching the aggregate gains reported in Table 5.

**Case B – Regulatory fine (−6.6%) on 24 Dec 2018.** On 24 December 2018, news surfaced that AAPL faced a substantial regulatory fine, triggering a 6.6% drop. Raw embeddings or a single-stage summary usually capture only high-frequency tokens such as fine and regulatory, overlooking the penalty's magnitude and downstream repercussions and thus under-reacting to the bearish signal. The hierarchical summary retains multiple detailed reports — including the fine's size and compliance commitments — in its first stage and condenses them in the second stage into a focused negative deep-dive summary. The PCA scatter plot (Fig. 3, upper-left quadrant) displays a distinct "in-depth + bearish" cluster, enabling the model to anticipate the sharp decline. Here, the hierarchical summary raises MCC by approximately 0.02, underscoring the robustness and stability gains delivered by our method.

![Figure 3. Interpretation of PCA quadrants: First quadrant (PC1>0, PC2>0): Major positive event + in-depth analysis → Strong bullish signal; Second quadrant (PC1>0, PC2<0): Lightweight positive news + brief overview → Weak bullish signal; Third quadrant (PC1<0, PC2>0): In-depth analysis of negative events → Strong bearish signal; Fourth quadrant (PC1<0, PC2<0): Brief negative news → Weak bearish signal.](images/fig3_pca_quadrants.png)
*Fig. 3. Interpretation of PCA quadrants: First quadrant of the coordinate axes (PC1 > 0, PC2 > 0): Major positive event + in-depth analysis → Strong bullish signal; Second quadrant of the coordinate axes (PC1 > 0, PC2 < 0): Lightweight positive news + brief overview → Weak bullish signal; Third quadrant of the coordinate axes (PC1 < 0, PC2 > 0): In-depth analysis of negative events → Strong bearish signal; Fourth quadrant of the coordinate axes (PC1 < 0, PC2 < 0): Brief negative news → Weak bearish signal.*

### 4.8. Hyperparameter study

We conducted a grid search over two Co-Attention Fusion Module hyperparameters: the number of fusion layers M and the number of attention heads per layer H. Fig. 4 reports ACC and MCC for all 16 (M, H) configurations on both benchmarks.

**CMIN-US results.** The highest accuracy (67.01%) is achieved at (M=2, H=16). The maximum MCC (0.350) occurs at (M=4, H=8), where ACC is 63.12%.

**CMIN-CN results.** The best accuracy (70.24%) is observed at (M=4, H=4). The peak MCC (0.299) is reached at (M=2, H=32), with ACC = 68.70%.

Based on these findings, we adopt (M=2, H=16) for CMIN-US and (M=4, H=4) for CMIN-CN to optimize accuracy. Alternative settings can be chosen if prioritizing MCC.

![Figure 4. ACC and MCC on the CMIN-US and CMIN-CN datasets as functions of the number of fusion layers (M) and attention heads per layer (H) in the Co-Attention Fusion Module.](images/fig4_acc_mcc_heatmaps.png)
*Fig. 4. ACC and MCC on the CMIN-US and CMIN-CN datasets as functions of the number of fusion layers (M) and attention heads per layer (H) in the Co-Attention Fusion Module.*

---

## 5. Discussion

This study proposed a unified framework that integrates financial news and historical price series via hierarchical language modeling, patch-based temporal encoding, and co-attention. On the CMIN-US benchmark (see Section 4.5), the model attained 67.01% accuracy and 0.346 MCC, and on CMIN-CN (see Section 4.5) it reached 70.24% accuracy and 0.293 MCC, outperforming both unimodal baselines and simpler fusion methods.

Detailed analysis yields four principal insights. First, the inclusion of textual features alongside price data produces consistent improvements of 5–9 percentage points in accuracy and 0.015–0.217 in MCC relative to price-only models, indicating that semantic information from news articles complements price fluctuations. Second, the multi-stage LLM summarization strategy exceeds direct embedding and single-stage approaches by up to +3.3 pp in accuracy and +0.022 in MCC, underscoring the importance of hierarchical semantic refinement under input-length constraints. Third, patch-based encoding preserves both local and global temporal structure while reducing attention complexity, thereby enabling efficient modeling of long price sequences. Fourth, co-attention achieves the most effective modality fusion, yielding up to +1.35 pp in accuracy and +0.045 in MCC over conventional cross-attention, which highlights the value of dynamic, fine-grained alignment between text and price representations. These results advance the understanding of joint representation learning by demonstrating how hierarchical semantic distillation and efficient temporal encoding can be combined with dynamic fusion within a single architecture.

Several limitations should be noted. First, this study relies on a general-purpose LLM that has not been fine-tuned on financial text, which may limit its ability to capture specialized market terminology. Second, the use of fixed patch sizes and strides may not generalize across assets with heterogeneous volatility profiles. Third, the evaluation is confined to the CMIN-US and CMIN-CN benchmarks, and the model's generalization to other markets or different temporal horizons remains to be investigated. Nonetheless, our experiments demonstrate that hierarchical summarization, patch-based encoding, and co-attention fusion collectively enhance multimodal stock-trend forecasting performance.

Future research should focus on fine-tuning the language model on sector-specific corpora to enhance its grasp of specialized terminology, devising adaptive patching schemes that account for asset-level volatility, integrating additional data sources — such as social-media sentiment and macroeconomic indicators — to improve robustness, and deploying the framework within a production-grade, real-time inference pipeline to rigorously evaluate its latency and scalability.

---

## 6. Conclusion

We propose a multimodal stock-trend prediction framework powered by large language models that offers a unified solution to two core challenges: extracting salient signals from lengthy financial texts and modeling the bidirectional dependency between textual semantics and price dynamics. The framework employs hierarchical summarization to distill news semantics, uses a patch-based temporal module to robustly encode the sequential structure of both prices and texts, and integrates the two modalities via text-price co-attention to form a richer, lower-noise joint representation. Cross-market benchmark experiments show consistent improvements over strong baselines in directional accuracy, risk-adjusted returns, and inference efficiency. Ablation studies further indicate that hierarchical summarization, patch-based temporal modeling, and co-attention each provide substantial gains, with their combination yielding the best performance. Overall, the results demonstrate that coupling task-aligned semantic distillation with efficient temporal modeling and explicit cross-modal interaction can materially enhance both the accuracy and practicality of multimodal financial forecasting.

---

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author(s) used ChatGPT (OpenAI) to refine the language and improve readability. After using this tool, the author(s) reviewed and edited all AI-generated content as needed and take(s) full responsibility for the content of the published article.

## CRediT authorship contribution statement

**Yuntao Zhang:** Software, Visualization, Methodology, Writing – original draft, Writing – review & editing.
**Zheng Dong:** Methodology, Writing – review & editing.
**Wenrui Xu:** Methodology, Writing – review & editing.

## Data availability

Data will be made available on request.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## References

Ahbali, N., Liu, X., Nanda, A., Stark, J., Talukder, A., & Khandpur, R. P. (2022). Identifying corporate credit risk sentiments from financial news. In A. Loukina, R. Gangadharaiah, & B. Min (Eds.), *Proceedings of the 2022 conference of the north american chapter of the association for computational linguistics: Human language technologies: Industry track* (pp. 362–370). Hybrid: Seattle, Washington + Online: Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.naacl-industry.40

Anbaee Farimani, S., Jahan, M. V., & Milani Fard, A. (2024). An adaptive multimodal learning model for financial market price prediction. *IEEE Access*, 12, 121846–121863. https://doi.org/10.1109/ACCESS.2024.3441029

Ansari, Y. (2024). Multi-cluster graph (MCG): A novel clustering-based multi-relation graph neural networks for stock price forecasting. *IEEE Access*, 12, 154482–154502. https://doi.org/10.1109/ACCESS.2024.3476159

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.

Bao, H., Wang, W., Dong, L., Liu, Q., Mohammed, O. K., Aggarwal, K., Som, S., Piao, S., & Wei, F. (2022). VLMO: Unified vision-language pre-training with mixture-of-modality-experts. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, & A. Oh (Eds.), *Advances in neural information processing systems* (pp. 32897–32912). Curran Associates, Inc. (vol. 35).

Bertrand, A., Eagan, J. R., & Maxwell, W. (2023). Questioning the ability of feature-based explanations to empower non-experts in robo-advised financial decision-making. In *Proceedings of the 2023 ACM conference on fairness, accountability, and transparency* FAccT '23 (pp. 943–958). New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/3593013.3594053

Bi, J., Chen, Z., Yuan, H., & Zhang, J. (2024). Accurate water quality prediction with attention-based bidirectional LSTM and encoder-decoder. *Expert Systems with Applications*, 238, 121807. https://doi.org/10.1016/j.eswa.2023.121807

Boudoukh, J., Feldman, R., Kogan, S., & Richardson, M. (2013). *Which News Moves Stock Prices? A Textual Analysis.* Technical Report, National Bureau of Economic Research.

Cao, Y., Tian, Z., Guo, W., & Liu, X. (2025). Mspatch: A multi-scale patch mixing framework for multivariate time series forecasting. *Expert Systems with Applications*, 273, 126849. https://doi.org/10.1016/j.eswa.2025.126849

Chaudhari, K., & Thakkar, A. (2023). Neural network systems with an integrated coefficient of variation-based feature selection for stock price and trend prediction. *Expert Systems with Applications*, 219, 119527. https://doi.org/10.1016/j.eswa.2023.119527

Chen, W., Jiang, M., Zhang, W.-G., & Chen, Z. (2021). A novel graph convolutional feature based convolutional neural network for stock trend prediction. *Information Sciences*, 556, 67–94. https://doi.org/10.1016/j.ins.2020.12.068

Chicco, D., & Jurman, G. (2020). The advantages of the matthews correlation coefficient (MCC) over f1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), 1–13.

Chiu, Y.-L., Gao, X., Liu, H.-C., & Zhai, Q. (2025). Financial literacy of chatGPT: Evidence through financial news. *Finance Research Letters*, 78, 107088. https://doi.org/10.1016/j.frl.2025.107088

Das, A., Kong, W., Leach, A., Mathur, S. K., Rajat, S., & Yu, R. (2023). Long-term forecasting with tiDE: Time-series dense encoder. *Transactions on Machine Learning Research.*

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019, Volume 1 (long and short papers)* (pp. 4171–4186). https://doi.org/10.18653/v1/N19-1423

Dezhkam, A., & Manzuri, M. T. (2023). Forecasting stock market for an efficient portfolio by combining XGBoost and hilbert-huang transform. *Engineering Applications of Artificial Intelligence*, 118, 105626. https://doi.org/10.1016/j.engappai.2022.105626

Dong, Z., Huang, X., Yuan, G., Zhu, H., & Xiong, H. (2021). Butterfly-core community search over labeled graphs. *Proceedings of the VLDB Endowment*, 14(11), 2006–2018.

Du, S., Li, T., Yang, Y., & Horng, S.-J. (2020). Multivariate time series forecasting via attention-based encoder-decoder framework. *Neurocomputing*, 388, 269–279. https://doi.org/10.1016/j.neucom.2019.12.118

Euna, J., Choi, H.-R., & LEE, H. C. (2020). Stock prediction using combination of BERT sentiment analysis and macro economy index. *Journal of The Korea Society of Computer and Information*, 25(5), 47–56. https://doi.org/10.9708/jksci.2020.25.05.047

Feng, F., Chen, H., He, X., Ding, J., Sun, M., & Chua, T.-S. (2019). Enhancing stock movement prediction with adversarial training. In *Proceedings of IJCAI-19* (pp. 5843–5849). https://doi.org/10.24963/ijcai.2019/810

Fernando, B., Habrard, A., Sebban, M., & Tuytelaars, T. (2013). Unsupervised visual domain adaptation using subspace alignment. In *Proceedings of ICCV 2013* (pp. 2960–2967). https://doi.org/10.1109/ICCV.2013.368

Ferreira, F. G. D. C., Gandomi, A. H., & Cardoso, R. T. N. (2021). Artificial intelligence applied to stock market trading: A review. *IEEE Access*, 9, 30898–30917. https://doi.org/10.1109/ACCESS.2021.3058133

Fu, K., Li, H., & Shi, X. (2024). An encoder-decoder architecture with fourier attention for chaotic time series multi-step prediction. *Applied Soft Computing*, 156, 111409. https://doi.org/10.1016/j.asoc.2024.111409

Gao, J., Ying, X., Xu, C., Wang, J., Zhang, S., & Li, Z. (2021). Graph-based stock recommendation by time-aware relational attention network. *ACM Transactions on Knowledge Discovery from Data*, 16(1). https://doi.org/10.1145/3451397

Ge, D., Dong, Z., Cheng, Y., & Wu, Y. (2024). An enhanced spatio-temporal constraints network for anomaly detection in multivariate time series. *Knowledge-Based Systems*, 283, 111169.

Gregnanin, M., Smedt, J. D., Gnecco, G., & Parton, M. (2025). Stock price time series forecasting using dynamic graph neural networks and attention mechanism in recurrent neural networks. In R. Meo, & F. Silvestri (Eds.), *Machine learning and principles and practice of knowledge discovery in databases* (pp. 357–373). Cham: Springer Nature Switzerland.

Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are zero-shot time series forecasters. In *Advances in neural information processing systems* (pp. 19622–19635). Curran Associates, Inc. (vol. 36).

Han, Y., Kim, J., & Enke, D. (2023a). A machine learning trading system for the stock market based on n-period min-max labeling using XGBoost. *Expert Systems with Applications*, 211, 118581. https://doi.org/10.1016/j.eswa.2022.118581

Han, Y., Kim, J., & Enke, D. (2023b). A machine learning trading system for the stock market based on n-period min-max labeling using XGBoost. *Expert Systems with Applications*, 211, 118581.

Haq, A. U., Zeb, A., Lei, Z., & Zhang, D. (2021). Forecasting daily stock trend using multi-filter feature selection and deep learning. *Expert Systems with Applications*, 168, 114444. https://doi.org/10.1016/j.eswa.2020.114444

Hasan, T., Bhattacharjee, A., Islam, M. S., Mubasshir, K., Li, Y.-F., Kang, Y.-B., Rahman, M. S., & Shahriyar, R. (2021). XL-sum: Large-scale multilingual abstractive summarization for 44 languages. In *Findings of the association for computational linguistics: ACL-IJCNLP 2021* (pp. 4693–4703). https://doi.org/10.18653/v1/2021.findings-acl.413

Htun, H. H., Biehl, M., & Petkov, N. (2023). Survey of feature selection and extraction techniques for stock market prediction. *Financial Innovation*, 9(1), 26. https://doi.org/10.1186/s40854-022-00441-7

Hu, G., Zhou, Z., Li, Z., Dong, Z., Fang, J., Zhao, Y., & Zhou, C. (2025). Multiscale transformers with contrastive learning for UAV anomaly detection. *IEEE Transactions on Instrumentation and Measurement*, 74, 1–15. https://doi.org/10.1109/TIM.2025.3571126

Huang, B., Kang, F., Li, J., & Wang, F. (2023). Displacement prediction model for high arch dams using long short-term memory based encoder-decoder with dual-stage attention considering measured dam temperature. *Engineering Structures*, 280, 115686. https://doi.org/10.1016/j.engstruct.2023.115686

Huang, W., Nakamori, Y., & Wang, S.-Y. (2005). Forecasting stock market movement direction with support vector machine. *Computers & Operations Research*, 32(10), 2513–2522. https://doi.org/10.1016/j.cor.2004.03.016

Jain, J. K., & Agrawal, R. (2024). FB-GAN: A novel neural sentiment-enhanced model for stock price prediction. In *Proceedings of the joint workshop of the 7th financial technology and natural language processing…* (pp. 85–93). Torino, Italia: Association for Computational Linguistics.

Jia, F., Wang, K., Zheng, Y., Cao, D., & Liu, Y. (2024). GPT4MTS: Prompt-based large language model for multimodal time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(21), 23343–23351. https://doi.org/10.1609/aaai.v38i21.30383

Jiang, J., Wu, L., Zhao, H., Zhu, H., & Zhang, W. (2023). Forecasting movements of stock time series based on hidden state guided deep learning approach. *Information Processing & Management*, 60(3), 103328. https://doi.org/10.1016/j.ipm.2023.103328

Lei, L. (2018). Wavelet neural network prediction method of stock price trend based on rough set attribute reduction. *Applied Soft Computing*, 62, 923–932. https://doi.org/10.1016/j.asoc.2017.09.029

Li, Q., Tan, J., Wang, J., & Chen, H. (2021). A multimodal event-driven LSTM model for stock prediction using online news. *IEEE Transactions on Knowledge and Data Engineering*, 33(10), 3323–3337. https://doi.org/10.1109/TKDE.2020.2968894

Li, Q., Wang, T., Li, P., Liu, L., Gong, Q., & Chen, Y. (2014). The effect of news and public mood on stock movements. *Information Sciences*, 278, 826–840. https://doi.org/10.1016/j.ins.2014.03.096

Li, S., Sun, Y., Lin, Y., Gao, X., Shang, S., & Yan, R. (2024a). Causalstock: Deep end-to-end causal discovery for news-driven multi-stock movement prediction. In *The thirty-eighth annual conference on neural information processing systems* (pp. 47432–47454).

Li, S., Sun, Y., Lin, Y., Gao, X., Shang, S., & Yan, R. (2024b). Causalstock: Deep end-to-end causal discovery for news-driven multi-stock movement prediction. In *Advances in neural information processing systems* (pp. 47432–47454). (vol. 37).

Li, S., Sun, Y., Lin, Y., Gao, X., Shang, S., & Yan, R. (2024c). Causalstock: Deep end-to-end causal discovery for news-driven multi-stock movement prediction. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, & C. Zhang (Eds.), *Advances in neural information processing systems* (pp. 47432–47454). Curran Associates, Inc. (vol. 37).

Li, Y., Wang, M., Guan, M. et al. (2025). SAPTSTA-AnoECG: A patchtst-based ecg anomaly detection method with subtractive attention and data augmentation. *Applied Intelligence*, 55, 184. https://doi.org/10.1007/s10489-024-05881-5

Lin, W.-C., Tsai, C.-F., & Chen, H. (2022). Factors affecting text mining based stock prediction: Text feature representations, machine learning models, and news platforms. *Applied Soft Computing*, 130, 109673. https://doi.org/10.1016/j.asoc.2022.109673

Liu, C., Xu, Q., Miao, H., Yang, S., Zhang, L., Long, C., Li, Z., & Zhao, R. (2025). TimeCMA: Towards LLM-empowered multivariate time series forecasting via cross-modality alignment. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(18), 18780–18788. https://doi.org/10.1609/aaai.v39i18.34067

Liu, H., Zhao, T., Wang, S., & Li, X. (2023). A stock rank prediction method combining industry attributes and price data of stocks. *Information Processing & Management*, 60(4), 103358. https://doi.org/10.1016/j.ipm.2023.103358

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024a). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics*, 12, 157–173. https://doi.org/10.1162/tacl_a_00638

Liu, R., Liu, H., Huang, H., Song, B., & Wu, Q. (2024b). Multimodal multiscale dynamic graph convolution networks for stock price prediction. *Pattern Recognition*, 149, 110211. https://doi.org/10.1016/j.patcog.2023.110211

Liu, R., Liu, H., Huang, H., Song, B., & Wu, Q. (2024c). Multimodal multiscale dynamic graph convolution networks for stock price prediction. *Pattern Recognition*, 149, 110211. https://doi.org/10.1016/j.patcog.2023.110211

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.

Long, J., Chen, Z., He, W., Wu, T., & Ren, J. (2020). An integrated framework of deep learning and knowledge graph for prediction of stock price trend: An application in chinese stock exchange market. *Applied Soft Computing*, 91, 106205. https://doi.org/10.1016/j.asoc.2020.106205

Luo, D., Liao, W., Li, S., Cheng, X., & Yan, R. (2023a). Causality-guided multi-memory interaction network for multivariate stock price movement prediction. In *Proceedings of the 61st annual meeting of the association for computational linguistics (Volume 1: Long papers)* (pp. 12164–12176). https://doi.org/10.18653/v1/2023.acl-long.679

Luo, D., Liao, W., Li, S., Cheng, X., & Yan, R. (2023b). Causality-guided multi-memory interaction network for multivariate stock price movement prediction. In *Proceedings of the 61st annual meeting of the association for computational linguistics* (pp. 12164–12176).

Ma, D., Yuan, D., Huang, M., & Dong, L. (2024). VGC-GAN: A multi-graph convolution adversarial network for stock price prediction. *Expert Systems with Applications*, 236, 121204. https://doi.org/10.1016/j.eswa.2023.121204

Ma, Y., Mao, R., Lin, Q., Wu, P., & Cambria, E. (2023a). Multi-source aggregated classification for stock price movement prediction. *Information Fusion*, 91, 515–528. https://doi.org/10.1016/j.inffus.2022.10.025

Ma, Y., Mao, R., Lin, Q., Wu, P., & Cambria, E. (2023b). Multi-source aggregated classification for stock price movement prediction. *Information Fusion*, 91, 515–528.

Mallat, S. G. (1989). A theory for multiresolution signal decomposition: the wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674–693. https://doi.org/10.1109/34.192463

Maqbool, J., Aggarwal, P., Kaur, R., Mittal, A., & Ganaie, I. A. (2023). Stock prediction by integrating sentiment scores of financial news and MLP-regressor: A machine learning approach. *Procedia Computer Science*, 218, 1067–1078. https://doi.org/10.1016/j.procs.2023.01.086

Nam, K., & Seong, N. (2019). Financial news-based stock movement prediction using causality analysis of influence in the korean stock market. *Decision Support Systems*, 117, 100–112. https://doi.org/10.1016/j.dss.2018.11.004

Nguyen, N., & Quanz, B. (2021). Temporal latent auto-encoder: A method for probabilistic multivariate time series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35, 9117–9125. https://doi.org/10.1609/aaai.v35i10.17101

Pan, Z., Jiang, Y., Garg, S., Schneider, A., Nevmyvaka, Y., & Song, D. (2024). S²IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting. In *Proceedings of the 41st international conference on machine learning* (pp. 39135–39153). PMLR (vol. 235).

Pang, B., Wei, W., Li, X., Feng, X., & Li, C. (2023). A representation-learning-based approach to predict stock price trend via dynamic spatiotemporal feature embedding. *Engineering Applications of Artificial Intelligence*, 126, 106849. https://doi.org/10.1016/j.engappai.2023.106849

Peng, Y., & Qi, J. (2019). CM-GANS: Cross-modal generative adversarial networks for common representation learning. *ACM Transactions on Multimedia Computing Communications and Applications*, 15(1). https://doi.org/10.1145/3284750

Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. W. (2017). A dual-stage attention-based recurrent neural network for time series prediction. In *Proceedings of IJCAI-17* (pp. 2627–2633). https://doi.org/10.24963/ijcai.2017/366

Qu, Z., Meng, Y., Hou, X., Chi, R., Ai, Y., & Wu, Z. (2025). Integrated energy short-term multivariate load forecasting based on patchTST secondary decoupling reconstruction for progressive layered extraction multi-task learning network. *Expert Systems with Applications*, 269, 126446. https://doi.org/10.1016/j.eswa.2025.126446

Rossi, A. G., & Utkus, S. (2024). The diversification and welfare effects of robo-advising. *Journal of Financial Economics*, 157, 103869. https://doi.org/10.1016/j.jfineco.2024.103869

Ruan, Y., Durresi, A., & Alfantoukh, L. (2018). Using twitter trust network for stock market analysis. *Knowledge-Based Systems*, 145, 207–218. https://doi.org/10.1016/j.knosys.2018.01.016

Sawhney, R., Agarwal, S., Wadhwa, A., & Shah, R. R. (2020). Deep attentive learning for stock movement prediction from social media text and company correlations. In *Proceedings of EMNLP 2020* (pp. 8415–8426). https://doi.org/10.18653/v1/2020.emnlp-main.676

Shen, D., Qin, C., Wang, C., Dong, Z., Zhu, H., & Xiong, H. (2021). Topic modeling revisited: A document graph-based neural network perspective. In *Advances in neural information processing systems* (pp. 14681–14693). Curran Associates, Inc. (vol. 34).

Tang, H., Dong, P., & Shi, Y. (2019). A new approach of integrating piecewise linear representation and weighted support vector machine for forecasting stock turning points. *Applied Soft Computing*, 78, 685–696. https://doi.org/10.1016/j.asoc.2019.02.039

Tang, Y., Han, K., Wang, Y., Xu, C., Guo, J., Xu, C., & Tao, D. (2022). Patch slimming for efficient vision transformers. In *2022 IEEE/CVF Conference on computer vision and pattern recognition (CVPR)* (pp. 12155–12164). https://doi.org/10.1109/CVPR52688.2022.01185

Tian, H., Zhang, X., Zheng, X., Zhang, Z., & Zeng, D. D. (2024). Graph representation learning of multilayer spatial-temporal networks for stock predictions. *IEEE Transactions on Computational Social Systems*, (pp. 1–14). https://doi.org/10.1109/TCSS.2024.3459945

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Proceedings of the 31st international conference on neural information processing systems* NIPS'17 (pp. 6000–6010). Curran Associates Inc.

Wang, H., Wang, T., & Li, Y. (2020). Incorporating expert-based investment opinion signals in stock prediction: A deep learning framework. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(01), 971–978. https://doi.org/10.1609/aaai.v34i01.5445

Wang, H., Xie, Z., Chiu, D. K. W., & Ho, K. K. W. (2024a). Multimodal market information fusion for stock price trend prediction in the pharmaceutical sector. *Applied Intelligence*, 55(1), 77. https://doi.org/10.1007/s10489-024-05894-0

Wang, J., Hu, Y., Jiang, T.-X., Tan, J., & Li, Q. (2023). Essential tensor learning for multimodal information-driven stock movement prediction. *Knowledge-Based Systems*, 262, 110262. https://doi.org/10.1016/j.knosys.2023.110262

Wang, M., Izumi, K., & Sakaji, H. (2024b). LLMFactor: Extracting profitable factors through prompts for explainable stock movement prediction. In *Findings of the association for computational linguistics: ACL 2024* (pp. 3120–3131). https://doi.org/10.18653/v1/2024.findings-acl.185

Wang, Z., Huang, B., Tu, S., Zhang, K., & Xu, L. (2021). Deeptrader: A deep reinforcement learning approach for risk-return balanced portfolio management with market conditions embedding. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(1), 643–650. https://doi.org/10.1609/aaai.v35i1.16144

Wu, L., & Zhang, Y. (2023). Attention-based encoder-decoder networks for state of charge estimation of lithium-ion battery. *Energy*, 268, 126665. https://doi.org/10.1016/j.energy.2023.126665

Xu, H., Chai, L., Luo, Z., & Li, S. (2020). Stock movement predictive network via incorporative attention mechanisms based on tweet and historical prices. *Neurocomputing*, 418, 326–339. https://doi.org/10.1016/j.neucom.2020.07.108

Xu, J., & Xu, L. (2017). Chapter three - information fusion. In *Integrated system health management* (pp. 101–156). Academic Press. https://doi.org/10.1016/B978-0-12-812207-5.00003-1

Xu, Y., & Cohen, S. B. (2018). Stock movement prediction from tweets and historical prices. In *Proceedings of the 56th annual meeting of the association for computational linguistics (Volume 1: Long papers)* (pp. 1970–1979). https://doi.org/10.18653/v1/P18-1183

Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., & Raffel, C. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. In *Proceedings of NAACL 2021* (pp. 483–498). https://doi.org/10.18653/v1/2021.naacl-main.41

Yang, Y., Zhang, C., Song, X., Dong, Z., Zhu, H., & Li, W. (2023). Contextualized knowledge graph embedding for explainable talent training course recommendation. *ACM Transactions on Information Systems*, 42(2). https://doi.org/10.1145/3597022

Ye, Y., Dong, Z., Zhu, H., Xu, T., Song, X., Yu, R., & Xiong, H. (2023). Mane: Organizational network embedding with multiplex attentive neural networks. *IEEE Transactions on Knowledge and Data Engineering*, 35(4), 4047–4061. https://doi.org/10.1109/TKDE.2022.3140866

Yoo, J., Soun, Y., Park, Y.-c., & Kang, U. (2021). Accurate multivariate stock movement prediction via data-axis transformer with multi-level contexts. In *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining* (pp. 2037–2045). https://doi.org/10.1145/3447548.3467297

Yu, J., Gao, X., Wang, T., Lu, H., Li, B., Zhai, F., Xue, B., & Meng, Z. (2025). A feature matching-based method for few-shot multivariate time series anomaly detection with symmetric patch mask siam transformer. *Engineering Applications of Artificial Intelligence*, 154, 110894. https://doi.org/10.1016/j.engappai.2025.110894

Yu, X., Chen, Z., & Lu, Y. (2023). Harnessing LLMs for temporal data - a study on explainable financial time series forecasting. In *Proceedings of EMNLP 2023: Industry track* (pp. 739–753). https://doi.org/10.18653/v1/2023.emnlp-industry.69

Zhang, B., Yang, H., & Liu, X.-Y. (2023). Instruct-finGPT: Financial sentiment analysis by instruction tuning of general-purpose large language models. arXiv:2306.12659.

Zhang, H., Chan, S., Qin, S., Dong, Z., & Chen, G. (2024a). Smde: Unsupervised representation learning for time series based on signal mode decomposition and ensemble. *Knowledge-Based Systems*, 301, 112369. https://doi.org/10.1016/j.knosys.2024.112369

Zhang, H., Yu, P. S., & Zhang, J. (2025a). A systematic survey of text summarization: From statistical methods to large language models. *ACM Computing Surveys*, 57(11). https://doi.org/10.1145/3731445

Zhang, J., Guo, L., Song, L., Gao, S., Hao, C., & Li, X. (2025b). PatchTCN: Patch-based transformer convolutional network for times series analysis. In *Proceedings of the 2024 3rd international symposium on computing and artificial intelligence* (pp. 1–9). https://doi.org/10.1145/3711507.3711508

Zhang, J., Huang, J., Jin, S., & Lu, S. (2024b). Vision-language models for vision tasks: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(8), 5625–5644. https://doi.org/10.1109/TPAMI.2024.3369699

Zhang, Q., Qin, C., Zhang, Y., Bao, F., Zhang, C., & Liu, P. (2022). Transformer-based attention network for stock movement prediction. *Expert Systems with Applications*, 202, 117239. https://doi.org/10.1016/j.eswa.2022.117239

Zhang, Q., Zhang, Y., Bao, F., Liu, Y., Zhang, C., & Liu, P. (2024c). Incorporating stock prices and text for stock movement prediction based on information fusion. *Engineering Applications of Artificial Intelligence*, 127, 107377. https://doi.org/10.1016/j.engappai.2023.107377

Zhao, Y., & Yang, G. (2023). Deep learning-based integrated framework for stock price movement prediction. *Applied Soft Computing*, 133, 109921. https://doi.org/10.1016/j.asoc.2022.109921

Zhen, K., Xie, D., & Hu, X. (2025). A multi-feature selection fused with investor sentiment for stock price prediction. *Expert Systems with Applications*, 278, 127381. https://doi.org/10.1016/j.eswa.2025.127381

Zheng, D., Liu, J., Li, R.-H., Aslay, C., Chen, Y.-C., & Huang, X. (2017). Querying intimate-core groups in weighted graphs. In *2017 IEEE 11th international conference on semantic computing (ICSC)* (pp. 156–163). https://doi.org/10.1109/ICSC.2017.80

Zheng, J., Xie, L., & Xu, H. (2025). Multi-resolution patch-based fourier graph spectral network for spatiotemporal time series forecasting. *Neurocomputing*, 638, 130132. https://doi.org/10.1016/j.neucom.2025.130132

Zhou, T., Niu, P., Wang, X., Sun, L., & Jin, R. (2023). One fits all: Power general time series analysis by pretrained LM. In *Proceedings of the 37th international conference on neural information processing systems* NIPS'23. Curran Associates Inc.
