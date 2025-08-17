# The Distributional View of LLM Post-Training — A 3-Part Series (2)

In the previous post, we have started talking about the distributional view of post-training - 
let's use this one to dive deeper to compare various inference-time strategies and post-training techniques.

## Part 2 — Post-Training Best Practice and Its Limits
Given the stochastic nature of LLMs, p₀ almost always assigns a non-zero probability to the desired answer for any prompt. One could either (i) sample an arbitrarily large number of times or (ii) embed the target answer directly as hints in the prompt, 
both of which would almost guarantee the desired output.

In that case, one might start to wonder ...

### Q4: Is inference-time strategy all you need?
**A:** To put it formally - if p₀\(y\|x\) > 0 for all correct completions, then with:
* Infinite samples
* Perfect verifier
* Optimal prompt hint

You can always recover the correct answer via search with the verifier, even without retraining.
* Prompting = conditioning p₀ into a higher-reward subspace.
* Perfect hint = deterministic correctness.

It is essentially the same reweighting of the output distribution toward reward, 
but applied at runtime with no weight updates as in post-training.

**Example (Codegen)**
- **Best-of-N with tests:** “`int_to_roman(n)`. 1≤n≤3999. Code only.”  
  - Sample 8
  - → run through unit tests (4/9/40/90/400/900)
  - → return the pass-max solution

**Example (Formatting Instruction Following)**
- **Self-verification:** “Return **valid JSON** with keys `{city, country}` for city=Paris.”
  - First attempt missing `country`
  - → validator rejects
  - → retry
  - → valid JSON

Now if inference time strategy is all you need, why is post-training even needed?

---

### Q5: Why is post-training needed? What does it actually do?
**A:** In theory, yes, inference-time strategy is all you need — but in practice, coverage gaps, verifier noise, and runtime cost make pure inference-time search too costly. 
Post-training emerges as the efficient approximation that amortizes the cost into weights. 

From a distributional lens, both post-training and inference-time strategies are simply different ways to steer p₀ toward the optimal policy p*. And here’s the provocation - if you can do it at inference, you can bake it into the model, for example:
* Thinking longer (e.g. with CoT prompting) → teach the model in post-training to generate reasoning traces and multiple candidates before deciding.
* Self-verification → fine-tune with reward-model feedback loops so the model learns to score and refine its own outputs.
* Tool-augmented inference → train on retrieval-augmented or tool-use traces so the model internalizes that.

Now let’s put them side by side and see where they align, and where they don’t.

| **Aspect** | **Post-Training** | **Inference-Time Strategies** |
|------------|-------------------|-------------------------------|
| **Mathematical Form** | Modifies model parameters so that pθ approx the optimal policy (p*) globally. | Adjusts p₀ → p' at decoding time so that p' approx p* for the current query. |
| **View in Distributional Terms** | “Bakes” the target distribution p* into the model’s weights. | Dynamically reweights p₀ toward p* per query. |
| **Expressivity (in theory)** | Can approximate any inference-time transformation, with right data, capacity and RL enviroment. | Can emulate many post-training effects without weight updates. |
| **Adaptivity** | Fixed after training; needs retraining to adapt to new p*. | Can adapt instantly to new objectives, constraints, or contexts. |
| **Computation Cost** | Paid once during training for most cases (except cases such as dynamic thinking). | Paid at every query; can be computationally heavy for large search/verification loops. |

With that, looks like post-training is important and impactful after all. Let's dive deeper next — out of the popular post-training approaches, which approaches generalize better?
