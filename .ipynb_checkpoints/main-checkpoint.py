import json
import os
import re
import time
from collections import Counter
from tqdm import tqdm
from openai import OpenAI

from config import (
    API_KEY,
    BASE_URL,
    MODEL_NAME,
    TEMPERATURE_DEBATER,
    TEMPERATURE_JUDGE,
    NUM_ROUNDS,
    DATA_PATH,
    LOG_PATH,
    RESULT_PATH,
    DEBUG,
    PRINT_EVERY_CALL,
    SAVE_AFTER_EACH_QUESTION,
    MAX_QUESTIONS,
)
from prompts import (
    DEBATER_A_PROMPT,
    DEBATER_B_PROMPT,
    JUDGE_PROMPT,
    DIRECT_QA_PROMPT,
    SELF_CONSISTENCY_PROMPT,
)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)
def dprint(msg):
    if DEBUG:
        print(msg, flush=True)
if not BASE_URL.startswith("http://") and not BASE_URL.startswith("https://"):
    raise ValueError(f"Invalid BASE_URL: {BASE_URL}")

    
def call_llm(system_prompt, user_prompt, temperature=0.7, tag="LLM"):
    start = time.time()
    dprint(f"[CALL START] {tag} | temp={temperature}")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.choices[0].message.content.strip()
    elapsed = time.time() - start

    if PRINT_EVERY_CALL:
        dprint(f"[CALL END] {tag} | {elapsed:.2f}s")
        preview = text[:180].replace("\n", " ")
        dprint(f"[PREVIEW] {tag}: {preview}...")

    return text

def extract_answer(text):
    if not text:
        return "UNKNOWN"

    # Final_Answer: same line
    m = re.search(r"final_answer:\s*(.+)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        if ans:
            return ans

    lines = text.splitlines()

    # Final_Answer: next non-empty line
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("final_answer:"):
            after = line.split(":", 1)[1].strip()
            if after:
                return after
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    return lines[j].strip()

    # Answer:
    for line in lines:
        if line.strip().lower().startswith("answer:"):
            after = line.split(":", 1)[1].strip()
            if after:
                return after

    return "UNKNOWN"

def direct_qa(question):
    out = call_llm(
        DIRECT_QA_PROMPT,
        f"Question: {question}",
        temperature=0.2,
        tag="DirectQA"
    )
    return {"raw": out, "answer": extract_answer(out)}

def self_consistency(question, n_calls):
    answers = []
    raws = []

    for i in range(n_calls):
        out = call_llm(
            SELF_CONSISTENCY_PROMPT,
            f"Question: {question}",
            temperature=0.8,
            tag=f"SelfConsistency_{i+1}/{n_calls}"
        )
        raws.append(out)
        answers.append(extract_answer(out))

    majority = Counter([canonical_label(a) for a in answers]).most_common(1)[0][0]

    return {"raw_samples": raws, "answer": majority}
def canonical_label(text):
    t = (text or "").strip().lower()

    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"

    negative_patterns = [
        "is not", "are not", "did not", "does not",
        "cannot", "can't", "not flat", "false", "refuted"
    ]
    positive_patterns = [
        "true", "supported"
    ]

    if any(p in t for p in negative_patterns):
        return "no"
    if any(p in t for p in positive_patterns):
        return "yes"

    return t.replace(".", "").strip()
def run_debate(question):
    transcript = []

    dprint("[DEBATE] Initial positions starting...")

    a_init = call_llm(
        DEBATER_A_PROMPT,
        f"Question: {question}\n\nDebate transcript so far:\nNone",
        temperature=TEMPERATURE_DEBATER,
        tag="DebaterA_Init"
    )

    b_init = call_llm(
        DEBATER_B_PROMPT,
        f"Question: {question}\n\nDebate transcript so far:\nNone",
        temperature=TEMPERATURE_DEBATER,
        tag="DebaterB_Init"
    )

    transcript.append({"round": 0, "A": a_init, "B": b_init})

    a_init_ans = canonical_label(extract_answer(a_init))
    b_init_ans = canonical_label(extract_answer(b_init))

    dprint(f"[DEBATE] Initial A = {a_init_ans}")
    dprint(f"[DEBATE] Initial B = {b_init_ans}")

    # Skip if both agree initially
    if a_init_ans == b_init_ans:
        dprint("[DEBATE] Consensus reached at initialization. Skipping rounds.")
    else:
        same_count = 0

        for r in range(1, NUM_ROUNDS + 1):
            dprint(f"[DEBATE] Starting round {r}/{NUM_ROUNDS}")

            running_text = ""
            for item in transcript:
                running_text += f"Round {item['round']}:\nA: {item['A']}\nB: {item['B']}\n\n"

            a_turn = call_llm(
                DEBATER_A_PROMPT,
                f"Question: {question}\n\nDebate transcript so far:\n{running_text}",
                temperature=TEMPERATURE_DEBATER,
                tag=f"DebaterA_Round{r}"
            )

            running_text += f"Round {r} partial:\nA: {a_turn}\n"

            b_turn = call_llm(
                DEBATER_B_PROMPT,
                f"Question: {question}\n\nDebate transcript so far:\n{running_text}",
                temperature=TEMPERATURE_DEBATER,
                tag=f"DebaterB_Round{r}"
            )

            transcript.append({"round": r, "A": a_turn, "B": b_turn})

            a_ans = canonical_label(extract_answer(a_turn))
            b_ans = canonical_label(extract_answer(b_turn))

            dprint(f"[DEBATE] Round {r} answers | A={a_ans} | B={b_ans}")

            if a_ans == b_ans:
                same_count += 1
            else:
                same_count = 0

            if same_count >= 2:
                dprint("[DEBATE] Early stopping triggered.")
                break

    full_text = ""
    for item in transcript:
        full_text += f"Round {item['round']}:\nA: {item['A']}\nB: {item['B']}\n\n"

    judge = call_llm(
        JUDGE_PROMPT,
        f"Question: {question}\n\nFull debate transcript:\n{full_text}",
        temperature=TEMPERATURE_JUDGE,
        tag="Judge"
    )

    judge_answer = extract_answer(judge)
    dprint(f"[JUDGE] Extracted answer = {judge_answer}")

    return {
        "transcript": transcript,
        "judge_raw": judge,
        "judge_answer": judge_answer,
    }

def normalize(x):
    return x.strip().lower().replace(".", "")


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if MAX_QUESTIONS is not None:
        data = data[:MAX_QUESTIONS]

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_logs = []

    correct_direct = 0
    correct_sc = 0
    correct_debate = 0

    total_debate_calls = 2 + (2 * NUM_ROUNDS) + 1

    dprint(f"[RUN] Loaded {len(data)} questions")
    dprint(f"[RUN] Debate-equivalent call budget for self-consistency = {total_debate_calls}")

    overall_start = time.time()

    for idx, item in enumerate(tqdm(data, desc="Questions"), start=1):
        q_start = time.time()

        q = item["question"]
        gt = item["answer"]

        dprint("\n" + "=" * 80)
        dprint(f"[QUESTION {idx}/{len(data)}] {q}")
        dprint(f"[GROUND TRUTH] {gt}")

        direct = direct_qa(q)
        sc = self_consistency(q, total_debate_calls)
        debate = run_debate(q)

        direct_pred = canonical_label(direct["answer"])
        sc_pred = canonical_label(sc["answer"])
        debate_pred = canonical_label(debate["judge_answer"])
        gt_label = canonical_label(gt)

        direct_ok = direct_pred == gt_label
        sc_ok = sc_pred == gt_label
        debate_ok = debate_pred == gt_label

        correct_direct += int(direct_ok)
        correct_sc += int(sc_ok)
        correct_debate += int(debate_ok)

        row = {
            "id": item.get("id", idx),
            "question": q,
            "ground_truth": gt,
            "ground_truth_label": gt_label,
            "direct_qa": direct,
            "self_consistency": sc,
            "debate": debate,
            "direct_pred": direct_pred,
            "sc_pred": sc_pred,
            "debate_pred": debate_pred,
            "correct_direct": direct_ok,
            "correct_self_consistency": sc_ok,
            "correct_debate": debate_ok,
        }

        all_logs.append(row)

        if SAVE_AFTER_EACH_QUESTION:
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(all_logs, f, indent=2, ensure_ascii=False)

        elapsed_q = time.time() - q_start
        dprint(f"[DONE QUESTION {idx}] {elapsed_q:.2f}s")
        dprint(f"[RESULT] Direct={direct_pred} ({direct_ok}) | SC={sc_pred} ({sc_ok}) | Debate={debate_pred} ({debate_ok})")

    summary = {
        "total": len(data),
        "direct_accuracy": correct_direct / len(data),
        "self_consistency_accuracy": correct_sc / len(data),
        "debate_accuracy": correct_debate / len(data),
        "model": MODEL_NAME,
        "num_rounds": NUM_ROUNDS,
        "runtime_seconds": round(time.time() - overall_start, 2),
    }

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nFINAL SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()