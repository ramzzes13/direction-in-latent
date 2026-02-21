# Bachelor's Thesis Research Plan: Automated Interpretation of Semantic Latent Directions in Diffusion Models

## 1. Introduction

### Problem Statement
Recent research, specifically the paper *Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models* (Haas et al.), has demonstrated that the internal representations (h-space) of Denoising Diffusion Models (DDMs) contain interpretable semantic directions. By applying unsupervised techniques like Principal Component Analysis (PCA) to these representations, one can discover vectors that control attributes such as pose, gender, or lighting.

However, a significant limitation of this unsupervised approach is that **the discovered directions are unlabeled**. The original authors state: *"their effects must be interpreted manually."* This means a human must generate images, visually inspect the changes (e.g., "this direction seems to rotate the face"), and manually assign a label. This process is subjective, time-consuming, and unscalable, especially when scaling to hundreds of directions or new datasets.

### Goal & Contribution
The goal of this bachelor's thesis is to **automate the interpretation of unsupervised semantic directions**. Instead of manual inspection, we propose to use pre-trained Vision-Language Models (VLMs), such as CLIP or BLIP, to automatically analyze and label the semantic shifts induced by h-space directions.

**Contribution:** A pipeline that takes a pre-trained Diffusion Model, discovers latent directions via PCA, and automatically outputs a textual label for each direction (e.g., "Principal Component 1: Rotation", "Principal Component 2: Smile").

## 2. Literature Review

### Foundations
-   **Original Work:** Haas et al. (2024) defined the *h-space* (U-Net bottleneck activations) and showed that PCA on this space yields disentangled edits. They relied on manual verification.
-   **Latent Semantics:** Earlier works in GANs (GANSpace, SeFa) also used PCA/matrix factorization but similarly relied on human interpretation or pre-defined attribute classifiers (InterfaceGAN) which require labeled data.

### Relevant Technologies
-   **CLIP (Contrastive Language-Image Pretraining):** A model that connects text and images. It can be used to measure the similarity between an edited image and a text prompt (e.g., "a smiling face").
-   **VLM Captioning (e.g., BLIP, LLaVA):** Models capable of answering questions about images or describing differences between two images.

### Gap
While there are methods to *find* directions (Haas et al.) and methods to *edit* with text (Prompt-to-Prompt), there is no standard automated method to *label* the directions found by unsupervised discovery in Diffusion Models.

## 3. Methodology

The proposed pipeline consists of three stages:

### Stage 1: Direction Discovery (Replication)
We will replicate the unsupervised method from the original paper:
1.  Load a pre-trained unconditional Diffusion Model (e.g., DDPM on CelebA-HQ).
2.  Generate $N$ random samples ($N \approx 500$).
3.  Collect the bottleneck activations $h_t$ for all timesteps $t$.
4.  Perform Incremental PCA on the collected activations to obtain the top $K$ principal components (eigenvectors $v_1, ..., v_K$).

### Stage 2: Edit Generation
For each discoverd direction $v_k$, we will generate visualization pairs to represent its effect:
1.  Sample a random seed latent $x_T$.
2.  Generate the "Original" image $I_{orig}$.
3.  Generate "Positive" and "Negative" edited images by injecting the direction:
    -   $I_{pos} = \text{Gen}(h + \alpha \cdot v_k)$
    -   $I_{neg} = \text{Gen}(h - \alpha \cdot v_k)$
    where $\alpha$ is a scaling factor (strength).

### Stage 3: Automated Labeling
We will test two approaches to label the transformation $(I_{neg} \to I_{pos})$:

**Approach A: CLIP-based Zero-Shot Classification**
-   Define a predefined list of potential attributes (e.g., ["smile", "glasses", "male", "female", "turn left", "turn right", "brightness", "zoom"]).
-   Measure the CLIP similarity change: $\Delta S = \text{CLIP}(I_{pos}, \text{text}) - \text{CLIP}(I_{neg}, \text{text})$.
-   Assign the label with the highest positive $\Delta S$.

**Approach B: VLM Difference Captioning**
-   Feed the pair $(I_{neg}, I_{pos})$ into a VLM (e.g., BLIP-2 or GPT-4o-mini).
-   Prompt: *"Describe the main difference between image 1 and image 2 in one or two words."*
-   Use the output as the open-ended label.

## 4. Experimental Design

This section describes exactly what needs to be implemented and measured.

### 4.1. Setup
-   **Dataset:** CelebA-HQ (Faces). This is the standard benchmark used in the original paper.
    -   *Why?* The semantic attributes (smile, gender, hair color) are well-defined and easy to verify.
-   **Model:** Pre-trained DDPM (from `diffusers` library or the original paper's codebase).
-   **Compute:** A single GPU (e.g., Colab T4 or local RTX 3060) is sufficient.

### 4.2. Experiments

#### Experiment 1: Validation of Direction Discovery (Replication)
*   **Objective:** Confirm we can reproduce the PCA directions from Haas et al.
*   **Procedure:** Run PCA on 500 samples. Visualize the top 5 components.
*   **Evaluation:** Visual comparison with Figure 3 of the original paper. Does PC1 look like "rotation/pose"? Does PC2 look like "gender/background"?

#### Experiment 2: CLIP-based Labeling Accuracy
*   **Objective:** Evaluate if CLIP can correctly identify the attribute.
*   **Procedure:**
    1.  Select top 10 PCA directions.
    2.  Manually label them to create a "Ground Truth" (e.g., Dir 1 = Rotation, Dir 2 = Smile).
    3.  Run the CLIP-based classification (Approach A) with a list of 20 common facial attributes.
    4.  Compare top-1 predicted label vs. Ground Truth.
*   **Metric:** Top-1 Accuracy (%).

#### Experiment 3: Open-Ended VLM Labeling
*   **Objective:** Test if modern VLMs can describe more subtle or complex directions that aren't in a fixed list.
*   **Procedure:**
    1.  Use the same top 10 directions.
    2.  Prompt a VLM (e.g., BLIP-2/LLaVA): *"What feature changes from the first image to the second?"*
    3.  Qualitatively evaluate the descriptiveness of the captions.
*   **Metric:** Qualitative Score (1-5) based on how well the caption matches the visual change.

#### Experiment 4: Sensitivity to Scale ($\alpha$)
*   **Objective:** Determine the optimal edit strength $\alpha$ for automated detection.
*   **Procedure:** Vary $\alpha$ from 1.0 to 10.0. Measure the confidence score of the CLIP classification.
*   **Hypothesis:** Too small $\alpha$ = no detectable change; too large $\alpha$ = image artifacts confuse the classifier.

### 4.3. Expected Results
-   **Exp 1:** We expect to recover the same primary directions (Rotation, Gender) as the original paper.
-   **Exp 2:** CLIP should achieve high accuracy (>80%) for distinct attributes like "Gender" or "Glasses" but might struggle with entangled directions (e.g., "Age" often changes "Glasses" too).
-   **Exp 3:** VLM descriptions might be more noisy but could capture combined effects (e.g., "The person becomes older and puts on glasses").

## 5. Discussion
-   **Limitations:** The method relies on the VLM's understanding. If the VLM is biased, the labels will be biased.
-   **Entanglement:** PCA directions are often entangled. The automated label might only pick up the most dominant feature (e.g., labeling "Gender change" simply as "Short hair").
-   **Generalization:** This thesis focuses on faces; future work could test it on LSUN Churches or Cars.

## 6. Conclusion
This thesis aims to bridge the gap between unsupervised discovery and practical usability. By automating the labeling of latent directions, we transform "interpretable" directions (which require human effort) into "self-describing" directions, making the latent space of diffusion models significantly more accessible for downstream applications.

## 7. References
1.  Haas, R., et al. (2024). *Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models*.
2.  Radford, A., et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision* (CLIP).
3.  Li, J., et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*.
4.  Härkönen, E., et al. (2020). *GANSpace: Discovering Interpretable GAN Controls*.
