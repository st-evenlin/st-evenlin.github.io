# The Distributional View of LLM Post-Training — A 3-Part Series (1)

When I started thinking about this topic, a lot of questions were floating around in my head, and I couldn't help but go through them one by one. I figured it might be interesting to present a Q&A series to take you through my journey of navigating this space.  

The Q&A series below moves from foundations to deployment, taking a practical, distributional lens on LLM post-training that aims to unify PPO/GRPO, DPO/ORPO, and inference-time strategies—with concrete examples.

## Background & Assumptions

- **Basics of LLM training:** This short series assumes a basic understanding of LLM pre- and post-training and common decoding strategies. 
- **Post-training best practices:** While not covered in this post, please refer to these excellent resources for modern-day LLM post-training best practices:
    - [Reinforcement Learning from Human Feedback: Progress and Challenges](https://www.youtube.com/watch?v=hhiLw5Q_UFg) by John Schulman
    - [Don't Teach, Incentivize](https://www.youtube.com/watch?v=kYWUEV_e2ss) by Hyung Won Chung
    - [Asymmetry of verification and verifier’s law](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) by Jason Wei


## Part 1 — Foundations

### Q1: What is LLM post-training, in the simplest possible terms?
**A:** Reshape the output distribution so the highest-reward completions become top-1.  

$$
p_\theta(y\mid x)\ \propto\ p_0(y\mid x) * \exp(\beta * R(y\mid x))
$$

- p₀ = base model; pθ = ideal model; R = reward (explicit or implicit); β = how far you allow mass to move (your KL budget)

**Example (Codegen)**  
- **Prompt:** “`is_palindrome(s)→bool`. Ignore case & non-alphanumerics. Output code only.”
  - **Before (p₀):** Reverse the string but forget to ignore spaces *(fails on tests with spaces)*.  
  - **After (pθ):** Remove spaces, normalize, and reverse the string for comparison *(passes the tests)*.  

**Example (Instruction Following)**  
- **Prompt:** “Answer only with the ISO date for ‘next Monday’ from 2025-08-13.”  
  - **Before (p₀):** “Next Monday is August 18, 2025.” *(extra words violate the format)*  
  - **After (pθ):** “2025-08-18” *(format rewarded; verbosity penalized)*

If post-training is “probability mass surgery,” the next question is: why does this matter in practice? Why is it important?

---

### Q2: Why is that important?
**A:** Pre-training often has **pass@1 ≪ pass@k**[^]; the model “knows” the answer but buries it. Post-training shifts mass so pass@1 ≈ pass@k.

**Example (Math, k=5).**  
- **Prompt:** “What’s the factorial of 5? (Respond with a number only)”  
  - **Before (5 samples):** `24, 24, 120, 25, 120` → pass@1 = 0.4, pass@5 = 0.6  
  - **After (5 samples):** `120, 120, 120, 120, 120` → pass@1 = 1.0, pass@5 = 1.0  
  - **Interpretation:** The correct completion already existed; we concentrated mass on it.

**Example (Codegen, k=5).**  
- **Prompt:** “`factorial(n)→int` (iterative). Raise `ValueError` for `n<0`. Code only.”
  - **Before (5 samples):** only 1/5 passes → pass@1 = 0.0, pass@5 = 0.2  
  - **After (5 samples):** all pass → pass@1 = 1.0, pass@5 = 1.0

OK, now **how** do different families actually implement that probability mass shift?

[^] *pass@1* is the probability the first (top-1) completion is correct; *pass@k* is the probability **at least one** of k samples is correct.


---

### Q3: What’s the mathematical connection between different post-training methods?
**A:** PPO/GRPO (RL), DPO/IPO/ORPO (pairwise), and RAFT/RSFT (selection) all push the policy toward the same reward-tilted target; they differ in how R is obtained and how proximity to p₀ is enforced.

**Example (Codegen for implementing slugify(str)).**  
- Pairwise (DPO/IPO/ORPO): Learn from chosen vs. rejected code; raise the margin of compliant regex + strip over naïve `.replace(" ", "-")`.  
- Selection (RAFT/RSFT/RSO): Sample from p₀, keep only test-passing snippets; MLE on the kept set makes compliant code the top mode.  
- On-policy RL (PPO/GRPO): Generate → run tests → reward by pass rate → update with a KL leash; top-1 passes more often while staying stylistically close to the reference.  
- Inference-time reranking: Sample K, test each, pick the highest-scoring; no training, higher latency.

**Example (Summarization with style constraint)** (e.g., “Two sentences, no emoji.”)  
- Pairwise (DPO/IPO/ORPO): Prefers concise, no-emoji summaries.  
- Selection (RAFT/RSFT/RSO): Train only on outputs that meet length/emoji rules.  
- On-policy RL (PPO/GRPO): Reward = (coverage + brevity + no-emoji); KL prevents drift.  
- Inference-time reranking: Sample 8, pick the best by a style/verifier judge.
