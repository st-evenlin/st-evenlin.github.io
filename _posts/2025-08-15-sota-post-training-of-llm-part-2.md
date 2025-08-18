# The Distributional View of LLM Post-Training — A 3-Part Series (2)

In the previous post, we started discussing the distributional view of post-training—let’s use this one to dive deeper and compare various inference-time strategies and post-training techniques.

## Part 2 — Post-Training Best Practices and Their Limits
Given the stochastic nature of LLMs, p₀ almost always assigns a non-zero probability to the desired answer for any prompt. One could either (i) sample an arbitrarily large number of times or (ii) embed the target answer directly as hints in the prompt, either of which would almost guarantee the desired output.

In that case, one might start to wonder…

### Q4: Are inference-time strategies all you need?
**A:** Formally—if p₀(y|x) > 0 for all correct completions, then with:
* Infinite samples
* A perfect verifier
* An optimal prompt hint

you can always recover the correct answer via search plus a verifier, even without retraining.
* Prompting = conditioning p₀ into a higher-reward subspace.
* Perfect hint = deterministic correctness.

This is essentially the same reweighting of the output distribution toward reward, but applied at runtime with no weight updates, unlike post-training.

**Example (Codegen)**
- **Best-of-N with tests:** “`int_to_roman(n)`. 1≤n≤3999. Code only.”  
  - Sample 8  
  - → run unit tests (4/9/40/90/400/900)  
  - → return the max-pass solution

**Example (Formatting Instruction Following)**
- **Self-verification:** “Return **valid JSON** with keys `{city, country}` for city=Paris.”
  - First attempt is missing `country`
  - → validator rejects
  - → retry
  - → valid JSON

Now, if inference-time strategy is all you need, why is post-training needed?

---

### Q5: Why is post-training needed? What does it actually do?
**A:** In theory, yes—an inference-time strategy can suffice. But in practice, coverage gaps, verifier noise, and runtime cost make pure inference-time search too expensive. Post-training is the efficient approximation that amortizes those costs into the weights. 

From a distributional lens, both post-training and inference-time strategies are simply ways to steer p₀ toward the optimal policy p*. And here’s the provocation: if you can do it at inference, you can bake it into the model. For example:
* Thinking longer (e.g., with CoT prompting) → teach the model in post-training to generate reasoning traces and multiple candidates before deciding.
* Self-verification → fine-tune with reward-model feedback loops so the model learns to score and refine its own outputs.
* Tool-augmented inference → train on retrieval-augmented or tool-use traces so the model internalizes that capability.

Now let’s put them side by side to see where they align—and where they don’t.

| **Aspect** | **Post-Training** | **Inference-Time Strategies** |
|------------|-------------------|-------------------------------|
| **Mathematical Form** | Modifies model parameters so that pθ approximates the optimal policy (p*) globally. | Adjusts p₀ → p' at decoding time so that p' approximates p* for the current query. |
| **View in Distributional Terms** | “Bakes” the target distribution p* into the model’s weights. | Dynamically reweights p₀ toward p* per query. |
| **Expressivity (in theory)** | Can approximate any inference-time transformation with the right data, capacity, and RL environment. | Can emulate many post-training effects without weight updates. |
| **Adaptivity** | Fixed after training; needs retraining to adapt to a new p*. | Can adapt instantly to new objectives, constraints, or contexts. |
| **Computation Cost** | Paid once during training in most cases (except scenarios like dynamic thinking). | Paid at every query; can be computationally heavy for large search/verification loops. |

With that, it looks like post-training is important and impactful after all. Let’s dive deeper next—among popular post-training approaches, which ones generalize better?
