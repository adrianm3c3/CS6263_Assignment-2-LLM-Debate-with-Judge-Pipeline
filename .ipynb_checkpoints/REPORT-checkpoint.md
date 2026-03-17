# Assignment 2 — LLM Debate with Judge Pipeline

## 1. Methodology

I implemented a multi-agent LLM debate system consisting of three roles:
- Debater A
- Debater B
- Judge

The pipeline follows four stages:
1. Initialization (independent answers from both debaters)
2. Multi-round debate
3. Final judgment
4. Evaluation against ground truth

Both debaters generate initial answers. If they agree, the system can skip further rounds. Otherwise, the agents debate for up to 3 rounds, with early stopping if both converge.

The same base model (`meta-llama/Llama-3.1-8B-Instruct`) is used for all roles, with role-specific prompts:
- Debaters use higher temperature (~0.7) for diversity
- Judge uses lower temperature (~0.2) for stability

---

## 2. Experiments

I compared three approaches:

### Direct QA
Single model response per question.

### Self-Consistency
Multiple independent samples (same number of calls as debate) with majority voting.

### Debate + Judge Pipeline
Two agents debate over multiple rounds, followed by a judge selecting the final answer.

All methods were evaluated on the same dataset for fairness.

---

## 3. Results

| Method | Accuracy |
|---|---|
| Direct QA | 0.67 |
| Self-Consistency | 0.72 |
| Debate + Judge | 0.66 |

### Observations

Self-consistency achieved the highest accuracy (0.72), outperforming both Direct QA (0.67) and the Debate pipeline (0.66).

This suggests that aggregating multiple independent samples from the same model is more effective than structured debate in this setup.

The debate pipeline did not outperform simpler baselines, likely due to shared biases between agents, since all roles used the same underlying model.

However, debate produced more interpretable reasoning because decisions were based on explicit argument exchange rather than a single response.

---

## 4. Analysis

The debate framework functioned correctly in terms of orchestration, logging, and structured reasoning. However, its performance did not exceed baseline methods.

A key limitation is that all agents used the same model, reducing diversity in reasoning. This caused Debater B to sometimes generate weak or artificial counterarguments instead of genuinely strong opposing views.

Another issue observed was answer formatting. Early outputs included full sentences instead of strict "Yes/No" answers, which required normalization for correct evaluation.

Despite lower accuracy, the debate system provides valuable interpretability. The full transcript allows inspection of reasoning steps and identification of where incorrect conclusions arise.

---

## 5. Prompt Engineering

I designed structured prompts to enforce consistency and improve parsing.

Debater prompts required:
- Answer (Yes/No)
- Reasoning
- Rebuttal

This ensured consistent output format and encouraged explicit argumentation.

The judge prompt enforced:
- strongest and weakest arguments from each side
- structured analysis
- final answer in Yes/No format
- confidence score
- winner selection

Additionally, prompts explicitly instructed agents to:
- stay focused on the exact question
- avoid drifting into related topics
- avoid unsupported claims

Lower temperature for the judge improved decision stability.

---

## Dataset

I used a subset of the BoolQ dataset consisting of 100 binary (Yes/No) questions.

This dataset was selected because it aligns well with the debate format and allows straightforward evaluation.

---

## Appendix: Full Prompts

### Debater A

You are Debater A in an LLM debate.
Argue for the strongest plausible answer to the exact question asked.

Rules:

Focus strictly on the question asked.

Do not switch to related but different questions.

Be concise and be evidence based.

Format:
Answer: Yes or No
Reasoning: ...
Rebuttal: ...


### Debater B
You are Debater B in an LLM debate.
Challenge Debater A by presenting the strongest plausible opposing interpretation.

Rules:

Focus strictly on the question asked.

Do not invent unsupported claims.

If the opposite side is weak, still present the strongest counterargument.

Format:
Answer: Yes or No
Reasoning: ...
Rebuttal: ...


### Direct QA

Answer the question directly and concisely.

Format:
Answer: Yes or No
Reasoning: ...


### Self-Consistency
Answer the question directly and concisely.

Format:
Answer: Yes or No
Reasoning: ...


### Judge

You are the Judge in an LLM debate pipeline.

Focus strictly on the exact question asked.
Do not substitute related questions.

Return EXACTLY this format:

Strongest_A: ...
Weakest_A: ...
Strongest_B: ...
Weakest_B: ...
Analysis: ...
Final_Answer: Yes or No
Confidence: 1, 2, 3, 4, or 5
Winner: A or B

