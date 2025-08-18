# The Distributional View of LLM Post-Training — A 3-Part Series (3)

In the previous post, we started discussing the best practices of post-training—let’s use this one to dive deeper to understand its limits and what to do when simpler approaches stop working.

## Part 3 — Advanced Topics - Limits and the Ultimate Recipe
We know that both inference-time strategies and post-training methods can bring pass@1 much closer to pass@k. 
We also know that, in theory, pass@k can be arbitrarily high if k is sufficiently large. 

### Q6: What is the limit of post-training then? When does it fail? 
**A:** In practice, pass@k is bounded by various constraints around k (e.g., inference resources/time). 
Hence, when the pre-trained policy is too poor, pass@k might be far lower than 1 due to a small k—thus limiting the performance of both inference-time strategies and post-training methods. 

In addition, again bounded by various constraints, we can only sample from the policy and apply reward reweighting during post-training; hence, sampling efficiency directly determines the efficiency of the post-training method—e.g., sampling only from low-reward regions will produce fewer signals useful for improving the policy.

If that happens, what do we do? One idea is to provide hints to the model—would that work?

---

### Q7: What do hints do? Will it generalize?
**A:** The idea is to provide hints as part of the prompt during post-training and hope the model learns to generalize to test prompts without hints. 

From a distributional lens, providing hints reshapes the policy distribution so that sampling becomes more efficient during post-training
(e.g., with no hints, it gets ~0 reward for all samples; with hints, it can generate some samples with positive rewards). 

However, generalization is not guaranteed. You learn p(y|x,h), not p(y|x). 
You could mix the data with and without hints to create a dropout effect, or rotate learning with and without hints to create a curriculum effect, 
but you are still learning toward something different and hoping it generalizes. 

Can we do better? Would prompt-agnostic tricks be better? 

---

### Q8: Are prompt-agnostic tricks better?
**A:** Here, the idea is to provide prompt-agnostic hints 
or an external tool that fetches hints when called—either as inference-time tricks or as part of post-training.

They often work better: they apply identically at train/test time, so you don’t worry about hints being unavailable at test time, 
and they apply across tasks, making them relatively easy to create and maintain.

**Example (Codegen)**
- **Hint provider as a tool:** “Solve problem 'Souvenirs': Amaru is buying souvenirs in a foreign shop. There are N types of souvenirs... (IOI'2025 q1)”  
  - Include `fetch_hints_if_avail()` as a tool during post-training
  - Reward solving the problem without the tool highest, then with the tool
  - Incentivize the model to answer directly but know its boundary for questions beyond its ability (requiring impractical k for pass@k) and fetch hints dynamically

**Example (Default style)**
- **Hints during both training/inference:** “Please provide a solution to this math question ...”  
  - Default math-style hints added to prompts during post-training and applied at inference time
  - Steer the model’s response to the ideal style needed for math questions (e.g., step-by-step solutions, proper formatting)

Now we know how to use both training and inference levers—what’s the ultimate recipe here?

---

### Q9: How to put all these together? What’s the ultimate recipe here?
**A (mini case: competitive programming codegen)**  
1) **Bake-in shift at training:** pairwise prefs (style/format/compliance) + selection on test-passing samples (RLVR) + reward to encourage dynamic thinking length for harder problems and tool use when necessary.  
2) **Lock-in at inference:** unit-test/execution + pref verifier for reranking; apply long thinking and fetch hints for difficult questions/poor initial responses.  
3) **Guardrails & fallback:** auto-repair minor violations; if unresolved, return the best candidate with a brief notice.

---

### Closing thoughts
Post-training, prompting, reranking, and verification are all probability mass surgery. The craft is *when* to move mass (train vs. inference), *how* to score it (tests/verifiers/preferences), and *how* to keep gains when tasks and distributions shift.
