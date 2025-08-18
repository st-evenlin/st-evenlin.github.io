# The Distributional View of LLM Post-Training — A 3-Part Series (3)

In the previous post, we have started talking about the best practices of post-training - 
let's use this one to dive deeper to understand its limits and what to do when simpler approaches stop working.

## Post 3 — Advanced Topics - Limits and the Ultimate Recipe
We know that both inference-time strategies and post-training methods can bring pass@1 much closer to pass@k. 
We also know that in theory, pass@k can be arbitrarily high if k is sufficiently large. 

### Q6: What is the limit of post-training then? When does it fail? 
**A:** In practice, pass@k is bounded by various constrains around k (e.g. inference resource/time). 
Hence, when the pre-trained policy is too poor, pass@k might be far lower than 1 with a small k - 
thus limitig the performance of both inference-time strategies and post-training methods. 

In addition, again bounded by various constrains, we can only sample from the policy and apply reward reweighting 
during post-training, hence the sampling efficiency directly determines the efficiency of the post-training method, 
e.g. sampling only from low-reward regions will produce fewer signals for improving the policy.

If that happens, what do we do? One idea is to provide hints to the model, would that work?

---

### Q7: What do hints do? Will it generalize?
**A:** Assuming the idea here is to provide hints as part of the prompt during post-training and hoping it learns to generalize to
test prompts without hints. 

From a distributional lens, providing the hints reshapes the policy distribution so that the sampling becomes more efficient in post-training
(e.g. with no hints, it gets ~0 reward for all of the samples and with hints, it is able to generate some samples with positive rewards). 

However, the generalization is not guaranteed. You learn p(y\|x,h), not p(y\|x). 
You could mix the data w and w/o hints to create a drop-out effect or rotate learning w and w/o hints to create a curriculum learning effect, 
but you are still learning towards something different and hoping it generalize. 

Can we do better? Would prompt-agnostic tricks be better? 

---

### Q8: Are prompt-agnostic tricks better?
**A:** Assuming the idea here is to provide prompt-agnostic hints 
or an external tool that fetches hints when called either as inference-time tricks or as a part of post-training.

They do work better - they apply identically at train/test so that you don't worry about it being unavailable on the test set, 
they also apply across tasks so that it is relatively easy to create and maintain.

**Example (Codegen)**
- **Hint provider as a tool:** “Solve problem 'Souvenirs': Amaru is buying souvenirs in a foreign shop. There are N types of souvenirs... (IOI'2025 q1)”  
  - Include fetch_hints_if_avail() as a tool during post-training
  - Reward solving the problem w/o tool the highest, then with tool
  - Incentize the model to answer directly but know its boundary for questions beyond its ablitiy (requiring impractical k for pass@k) and fetch hints dynamically

**Example (Default style)**
- **Hints during both training/inference:** “Please provide a solution to this math question ...”  
  - Default math style hints added to prompts during post-training and applied during inference time
  - Steer the model response to the ideal style needed for math questions (e.g. with step-by-step solutions, proper formatting)

Now we know how to use both training and inference levers, what’s the ultimate recipe here?

---

### Q9: How to put all these together? What’s the ultimate recipe here?
**A (mini case: competitive programming codegen)**  
1) **Bake-in shift at training:** pairwise prefs (style/format/compliance) + selection on test-passing samples (RLVR) + reward to encourage dynamic thinking length for harder problems and tool use when necessary.  
2) **Lock-in at inference:** unit-test/execution + pref verifier for reranking; apply long thinking and fetch hints for difficult questions/poor initial responses.  
3) **Guardrails & fallback:** auto-repair minor violations; if unresolved, return best candidate with a brief notice.

---

### Closing thoughts
Post-training, prompting, reranking, and verification are all probability mass surgery. The craft is *when* to move mass (train vs inference), *how* to score it (tests/verifiers/preferences), and *how* to keep gains when tasks and distributions shift.
