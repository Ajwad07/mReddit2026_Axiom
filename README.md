# Axiom@mReDDIT2026: A Cost-Sensitive Learning Approach to Multilingual Regret Detection and Domain Identification

Official code repository for **Team Axiom**'s system submission to the **mReDDIT @ FIRE 2026** shared task: *Multilingual Regret Detection and Domain Identification from Social Media Text*.

---

## 🏆 Official Leaderboard Results

Our system decoupled the shared task into independent subtasks and applied cost-sensitive learning to handle severe class imbalance across high-resource and low-resource settings:

| Track | Task | Accuracy | Macro F1 | Weighted F1 | Leaderboard Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English** | Subtask A (Regret Detection) | **0.8346** | **0.8335** | **0.8341** | **3rd Place Overall** 🥉 |
| **English** | Subtask B (Domain Identification) | **0.8852** | **0.7646** | **0.8845** | — |
| **Tulu** | Subtask A (Regret Detection) | **0.6336** | **0.5591** | **0.6278** | **7th Place Overall** |
| **Tulu** | Subtask B (Domain Identification) | **0.7737** | **0.5430** | **0.7689** | — |

---

## 💡 System Architecture & Methodology

```text
                   ┌─────────────────────────────────────────┐
                   │    Input Text: "Title - Body Text"      │
                   └────────────────────┬────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            [English Track]                           [Tulu Track]
         xlm-roberta-base (L=512)               google/muril-base-cased (L=256)
                    │                                       │
          ┌─────────┴─────────┐                   ┌─────────┴─────────┐
          ▼                   ▼                   ▼                   ▼
    [Subtask A]         [Subtask B]         [Subtask A]         [Subtask B]
     3-Class             5-Class             3-Class             5-Class
  Regret Head         Domain Head         Regret Head         Domain Head
          │                   │                   │                   │
          └─────────┬─────────┘                   └─────────┬─────────┘
                    │                                       │
                    ▼                                       ▼
        Custom Weighted CrossEntropy             Custom Weighted CrossEntropy
        (Class Weight W_c = N / C*N_c)           (Class Weight W_c = N / C*N_c)
````
1. **Light-Touch Preprocessing & Text Concatenation:**
   * Preserved all punctuation, capitalization, and stopwords to retain crucial causal and syntactic cues.
   * Concatenated post title and post body (`Title - Text`) to maximize contextual signal within sequence bounds.
2. **Task Decoupling:**
   * Trained two separate single-task heads (Subtask A for Regret, Subtask B for Domain) to eliminate gradient interference between competing objectives.
3. **Language-Specific Encoders:**
   * **English Track:** Fine-tuned `xlm-roberta-base` for 15 epochs with sequence length $L=512$.
   * **Tulu Track:** Fine-tuned `google/muril-base-cased` (MuRIL) for 20 epochs with sequence length $L=256$ to leverage region-specific pre-training on Indic corpora.
4. **Cost-Sensitive Class Balancing:**
   * Subclassed Hugging Face `Trainer` to inject a dynamically computed class-weighted Cross-Entropy loss ($W_c = \frac{N}{C \times N_c}$).
   * Evaluated and saved best model checkpoints strictly on validation **Macro F1**.

---
