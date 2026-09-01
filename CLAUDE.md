# Working rules for this repository

This is a master's thesis. Its subject is a **re-implementation of a published
paper**, and its value depends entirely on the reader being able to tell which
claims come from the paper, which come from measurement, and which are
judgement calls. Everything below exists to protect that distinction.

Read `README.md` section 9 for the layout and `docs/ASSUMPTIONS.md` for every
judgement call made so far.

Rules 1 to 8 protect that distinction. Rules 9 and 10 are about explaining, and
rule 10 is the longest here because it applies most often: it sets how anything
is explained in this project, in a README or in a reply in conversation.

---

## The paper

Wen, J.-R., et al. (2026). *Explainable spatiotemporal graph neutral network
with dynamic SHAP enhancement for aeroengine RUL prediction.* Aerospace Science
and Technology, 168, 110875.

The PDF is at
`D:\Main\University\Thesis\Papers\Predictive Maintanace Papers\GNN\GNN Predictive Maintenance\RUL\MAPSS\Explainable Spatio-Temporal Graph Neural Network with Dynamic SHAP Enhancement for Aeroengine RUL Prediction.pdf`

No source code was published with it, and its data availability statement reads
"Data will be made available on request".

---

## Rules

### 1. Never state a fact about the paper from memory

Open the PDF and check. This applies to every table value, every equation, every
figure and every sentence you intend to quote.

Figures need care: a value read from a page thumbnail is a guess. Render the
region at high resolution (`pypdfium2`, scale 8 or more) before reading a number
off an axis, and say how you obtained it. The value of `RUL_max` in this project
was settled by pixel-calibrating a plateau against its own axis ticks, and the
paper's schematic figure disagrees with its experimental one.

If the paper prints something twice — the adjacency matrix appears in Table 3
and again in Fig. 3 — check both. They have agreed so far; that is a finding,
not an assumption.

### 2. Never report a number you have not computed

No estimated results, no "approximately", no numbers carried over from an
earlier message without re-deriving them. If a claim needs a number, run the
code that produces it and show the output.

Numbers already in `docs/ASSUMPTIONS.md` and `README.md` have been measured.
Numbers anywhere else have not, until you check.

### 3. One seed is not a result

Anything quoted as a result of this project is a mean over at least five seeds,
with its standard deviation:

```bash
python scripts/train.py --seeds 5
```

Before saying that one configuration beats another, test whether the difference
survives the seed-to-seed spread. A-21 was settled that way and turned out not
to be significant, which is a real answer and was worth having.

Then remember it may not stay settled. All three of this project's
configuration comparisons — A-21, A-02 and F-04 — were re-measured after the
A-07 correction of 2026-08-28, and two of them **reversed direction**. A
comparison is a fact about one architecture, not about the two options. Name
the configuration whenever you quote one, and re-run it whenever the
architecture moves.

The seed varies the train/validation split as well as the initialisation and
the batch order, so the spread is what a genuine repetition would show.
`--fixed-split` holds the split constant, which understates it; if you use that
flag, say so wherever the number appears.

### 3a. A stored result must name the code that produced it

Every run writes its commit — with a `-dirty` suffix when the tree has
uncommitted changes — into `summary.json` and into its own directory name,
alongside a UTC timestamp. Never hand-edit a results file, never quote a number
from a run whose commit you cannot identify, and never let two runs share a
directory.

This rule exists because a results file once survived here that contradicted
the documentation and could not be reproduced from any committed state. It
happened to contradict a figure that was correct. It could as easily have
corroborated one that was wrong. See F-09.

### 4. Every deviation from the paper gets a numbered entry

If the paper does not specify something and you have to choose, add an entry to
`docs/ASSUMPTIONS.md` with:

- **The gap** — quote what the paper does and does not say.
- **What we chose.**
- **Why** — the reasoning, not just the conclusion.
- **How much it matters** — and if that is an argument rather than a
  measurement, say so.

Then put the identifier in the code at the point of use, as a comment beginning
`ASSUMPTION A-xx`. The header of that file promises this; keep it true.

That comment must say **what the choice was, in words**, not just cite its
number. `# ASSUMPTION A-17` tells a reader nothing; `# ASSUMPTION A-17: shrink
the learning rate by the same factor every epoch. The paper gives the first and
last rate but never how it gets between them.` tells them everything they need
without leaving the file.

Observations *about the paper* established from data are findings (`F-xx`), not
assumptions. Keep the two apart: assumptions are our choices, findings are
facts about the source.

### 5. Distinguish the three kinds of statement, always

- **"The paper says X"** — verifiable in the PDF; cite the table or equation.
- **"We chose Y"** — an assumption; it needs an `A-xx` entry.
- **"We measured Z"** — a result; it needs the command that produced it.

Never let one wear the clothes of another. "This reproduces the paper" is a
claim about all three at once and is almost always wrong here: the LSTM stage is
not implemented, so results compare against the paper's own **"ESTGNN w/o
LSTM"** row (14.93 RMSE on FD001), never its headline figures.

### 6. Correct errors plainly, including your own

Several claims in this repository were wrong when first written and were fixed
after checking: F-05 was written from 14 sampled windows out of 240 and had both
its magnitude and its mechanism wrong. A-10 said "the only sensor" when seven
qualified. A-08 was logged as an assumption when Fig. 1 states it outright.

When you find one, fix it, say what was wrong and how it was found, and leave a
note if the error is instructive. Do not quietly overwrite. A corrected record is
worth more than a clean one.

### 7. Generated data is not allowed

The project deliberately contains no synthetic data. A pipeline that appears to
work on invented signals says nothing about whether it works on engines. If the
NASA files are missing, the code says so and points at the download; it never
substitutes something with the right shape.

### 8. Do not commit or push without being asked

Make the changes, run the checks, report what changed, then stop. Committing is
the author's decision, not yours.

### 9. Explanation lives in the folder READMEs, not in the code

Every folder under `src/thesis_gnn_rul_xai/` has a `README.md`, and that is
where the reasoning, the traps, the vocabulary and the meaning of every
argument belong. A `.py` file carries **one line** of description per module,
class and function, plus the `ASSUMPTION` comments required by rule 4.

Do not put the explanation back into the code. No `Parameters` / `Returns` /
`Raises` / `Notes` / `Examples` sections, and no Sphinx cross-reference roles
(`:class:`, `:func:`, `:meth:`, `:attr:`, `:data:`) — this project has no Sphinx
build, so they render as noise to everyone who reads them.

Write for someone who does not already know the project. Spell a term out the
first time it appears rather than assuming it. Where a README has to name a
class, say what it is: "a `TrainingHistory`, the record of a finished run", not
just the name.

#### The check, which is not the same as the rule

The paragraph above has been in this file from the start and was broken anyway,
twice. Stating the rule again does not work. Before calling any document,
comment or explanation finished, **run this check**:

> Re-read every sentence as one specific person: they know Python, they have
> never used PyTorch, and they have never read the paper. Any word that person
> would have to look up gets replaced, or explained in the same sentence.

The words that fail are the ones that feel ordinary from inside the code
because they were borrowed from PyTorch or from graph theory: *buffer*,
*broadcast*, *state dictionary*, *logits*, *self-loop*, *ablation*,
*hyperparameter*, *epoch*, *z-score*, *Glorot*. Invented words are worse still
— "an unrescaled table is rejected" survived here for weeks and means nothing
to anyone. `python scripts/check_prose.py` catches this specific list; it
cannot judge whether writing is clear, so it is a floor and not a substitute
for the check above.

Second half of the check: **a claim about how something works needs a worked
example with real numbers.** "A column of the weight matrix is a list of
multipliers" explains nothing on its own. Load the real data, print the actual
values, do the arithmetic term by term, and show the code producing the same
answer. Placeholders like `x1 ... x32` are not an example.

Worked examples are tests, in `tests/test_examples.py` and
`tests/test_examples_torch.py`, not doctests. When you add a function, add its
example there.

### 10. Explanation style

Rule 9 says where explanation lives. This says how to write it. **It governs
answers given in conversation as much as files in the repository** — a reply in
chat is subject to the same standard as a README, because it is where most
explaining actually happens.

#### 10.1 The objective

The objective of a conceptual or technical answer is not to be correct. It is
to leave the reader able to **reconstruct the answer themselves**. An answer
that is technically right and skips the step that makes it make sense is a bad
answer here, however accurate and however short.

Prioritise understanding over brevity. Brevity is never bought by dropping a
step.

Advanced vocabulary in the question is not evidence that the ground underneath
it is solid. Someone can ask a precise question about Adam's second moment
estimate and still be unsure what quantity is being differentiated and with
respect to what. Answer the question that was asked, and put the missing rung
back in the ladder on the way past.

The general rules that follow apply to every explanation:

- Assume the reader may not know the intermediate concepts, even when they know
  the general topic.
- Do not open with the implementation, the equation or the jargon. Start with
  the intuition and the problem the concept solves.
- Explain the mechanism step by step, and do not skip intermediate steps.
- Explain **why** each step works, not only what happens.
- If a concept depends on another, explain that prerequisite first, briefly.
- Explicitly separate concepts that are easy to confuse. Name both and say what
  distinguishes them.
- Use small concrete numerical examples wherever one is possible.
- When an equation appears, say what every symbol in it is and what the whole
  thing is doing.
- Finish with a short example of the concept in use.
- Do not assume that fluency with a term implies fluency with the mathematics
  under it.
- Avoid "simply", "obviously" and "as you know" unless the thing really has been
  established earlier in the same explanation.
- If a question has more than one reading, name the readings and say which one
  is being answered.
- Aim at one clear mental model, not a list of definitions.

#### 10.2 Four kinds of question, answered in four different orders

Work out which one is being asked before answering.

1. **A coding task.** Do the work. Explain what the relevant code does where
   understanding it is needed to judge the change.
2. **A conceptual question** — *why*, *how*, *what happens*, *what is this*.
   Teaching comes first and code comes last, if at all.
3. **A debugging question.** The cause first, the fix second (10.14).
4. **A research or design question.** The trade-offs first, the recommendation
   after them, and the basis for the recommendation named (10.16).

A question that starts with *why*, *how*, *what exactly happens* or *where does
this number come from* is a request for a **mechanism**. Do not answer it with
another definition, and do not answer it by dumping code.

> Bad: "Attention lets the model focus on the important nodes."
>
> Good: "Each node computes a compatibility score against each of its
> neighbours. The scores are turned into weights that add up to one. The node
> then adds up its neighbours' representations in those proportions, so a
> neighbour with weight 0.7 moves the result far more than one with weight
> 0.05."

Then show the arithmetic with numbers.

#### 10.3 Never skip the conceptual bridge

If the reasoning runs A → B → C → D, never explain A → D. Before writing a
conclusion, ask: *what intermediate idea has to be in place for this sentence
to mean anything?* Write that idea down first.

These phrases are where the missing step usually hides:

- "therefore", "obviously", "simply", "as expected"
- "this allows the model to …"
- "the network learns …"
- "the gradient is propagated …"
- "attention captures …"
- "the embeddings encode …"
- "the optimiser adjusts …"

Each one promises a mechanism and delivers a label instead. They are allowed
only when the next sentence says what actually happens, with the quantities
named: what is computed, from what, and what changes as a result.

#### 10.4 Assume the prerequisite is missing

If a question rests on something the reader may not have, teach that first,
briefly, then climb.

- Asked *how Adam uses the gradient*: do not open on first and second moment
  estimates. Establish what the loss is, what a gradient is, what is being
  differentiated with respect to what, what the sign and the size of a gradient
  mean, and why the parameters have to change at all. Then introduce Adam.
- Asked about **attention in a graph attention network**: establish how a node
  collects information from its neighbours at all, before any attention
  coefficient appears.
- Asked about **SHAP**: establish what it means for one feature to contribute
  to one prediction, and the cooperative-game picture it borrows, before
  Shapley values.
- Asked about **layer-wise relevance propagation**: establish what it means to
  split a prediction into shares and hand them backwards, before any
  propagation rule.

#### 10.5 Name the confusions

Whenever two ideas are routinely mixed up, name both and say what separates
them, in a short "**Do not confuse this with …**" note. The pairs that come up
in this project:

- gradient vs gradient descent; gradient vs derivative
- loss vs error; training loss vs test loss
- a parameter the model learns vs a setting chosen before training
- feature vs node feature; a learned node representation vs position in a
  sequence
- adjacency matrix vs node feature matrix
- sending information to a neighbour vs combining what arrived
- attention vs the neighbourhood that attention is computed over
- graph attention network vs graph transformer
- spatial correlation vs temporal correlation
- temporal dependency vs position in the sequence
- hidden state vs cell state; recurrent network vs LSTM vs GRU
- raw output scores vs probabilities; score vs predicted class
- explanation vs feature importance; local vs global explanation
- model-agnostic vs model-specific explanation
- a SHAP value vs an attention weight
- an explanation of a model vs a causal claim about engines
- correlation vs causation; interpolation vs extrapolation
- overfitting vs underfitting
- rescaling to a fixed range vs subtracting the mean and dividing by the spread
- validation set vs test set

The last one is settled here and is not a matter of taste: validation error and
test error are not comparable in this project (A-06).

#### 10.6 Small numbers — and which numbers may be invented

Prefer a tiny worked example to an abstract statement, always. One parameter
for a gradient. Three or four nodes for a graph. Two or three neighbours for
attention. Two to four features for an explanation method. Small enough that
every number can be checked by hand.

Do not write `y = Wx + b` and stop. Write:

```
x = [2, 3]        W = [[1, 0], [-1, 2]]        b = [0.5, -0.5]

Wx = [1*2 + 0*3,  -1*2 + 2*3] = [2, 4]
y  = [2 + 0.5,  4 - 0.5]      = [2.5, 3.5]
```

This is where this rule meets rules 2 and 7, and the difference decides whether
a number may be invented:

- **Illustrating an idea** — invent them freely, keep them tiny. The arithmetic
  above is teaching, and claims nothing about anything.
- **Saying what this project does or produces** — run the code and quote the
  output (rule 2). Never invent, never estimate, never carry a number over from
  an earlier message without re-deriving it.
- **Anything entering the repository** — worked examples use the real FD001
  files and live in `tests/test_examples.py` (rule 9), and invented sensor data
  never enters this project at all (rule 7).

A made-up number may illustrate arithmetic in a reply. It may never stand in
for a measurement, and it may never become a file.

#### 10.7 Show where the information goes

Trace the path, and say what happens at each stage rather than naming it. For a
network:

```
input → preprocessing → representation → layer → intermediate representation
      → prediction → loss → gradient → parameter update
```

For a graph network:

```
node features → graph structure → which neighbours may send → messages
      → combining what arrived → updated node representation → prediction
```

For an explanation method:

```
trained model → prediction → the method's probing (perturbing the input,
      reading gradients, dividing up relevance, or trying subsets of features)
      → per-feature contribution → interpretation
```

Naming those stages is not the explanation. Saying what enters each one, what
comes out, and what shape it has when it leaves, is.

#### 10.8 Equations get six things, or they do not get written

1. The equation.
2. Every symbol defined, including the indices.
3. What it computes, in plain language.
4. Why it is needed at all.
5. A tiny numerical example run through it.
6. How its result is used by the next step.

`h_i' = σ(Σ_j α_ij W h_j)` written and left alone explains nothing. What is
`h_i`; what is `h_j`; who is `j`, and why are neighbours involved; what does
`W` do to a vector; where does `α_ij` come from; why is the sum weighted at
all; what does `σ` do to the result; and what comes out at the end.

#### 10.9 Code: line → mathematics → purpose

Explaining machine-learning code is not translating Python into English. Map
each important line to the operation it performs and to the reason it is there.

```python
x = model(data)          # the forward pass: produces the prediction
loss = criterion(x, y)   # how far that prediction is from the target
loss.backward()          # computes d(loss)/d(parameter) for every parameter
optimizer.step()         # changes the parameters using those derivatives
```

`backward()` does not "just calculate the gradients": say what is being
differentiated, through which computations, and with respect to what.

When asked about **one line, answer that line first** — not the file around it
— in this order: plain English, an example input, the resulting value, why it
is used here, and any edge case. For `range(0, n_rows - window_size + 1,
stride)`, "it generates indices" is not an answer:

```
n_rows = 10, window_size = 4, stride = 2   →   [0, 2, 4, 6]
```

and then what each of those four numbers is the start of, and why the last
window has to begin at row 6 rather than row 8.

**Loops get their first two or three iterations written out with real values.**
This matters most for sliding windows, sequence building, graph construction,
batching, and perturbing inputs for an explanation method.

**Never assume an array is self-evident.** Say which of these it is — a single
number, a Python list, a tuple, a NumPy array, a PyTorch tensor — and give its
type, its shape and a sample of its contents:

```
starts = [0, 2, 4, 6]    # a Python list, length 4,
                         # each element the first row index of one window
```

**A data pipeline is explained by tracing it**, with the shape at each stage:
raw file → table → selected columns → array → windows → inputs and targets →
batch → model.

**A window is drawn, not asserted:**

```
sequence   [A B C D E F G H]        window size 4, stride 2

           [A B C D]
                 [C D E F]
                       [E F G H]
```

and then what the starting indices 0, 2 and 4 mean in the surrounding
algorithm.

#### 10.10 Gradients, backpropagation and optimisers

Whenever gradients come up, answer all of these explicitly:

- **What changes?** Usually the parameters.
- **What is held fixed?** The training example and its target, during that one
  calculation.
- **What quantity is differentiated?** The loss.
- **With respect to what?** The trainable parameters.
- **Why does a gradient exist here at all?** Because every step from parameter
  to loss is differentiable with respect to those parameters.

Keep apart a discrete **input**, a discrete **operation inside the model**, and
the **continuous parameters**. "Gradient descent cannot work with discrete
data" is false as a blanket statement, and saying it is a 10.3 failure.

For backpropagation, "the gradient propagates backward through the network" is
nearly useless on its own. Show the path

```
parameter → intermediate computation → prediction → loss
```

and then obtain d(loss)/d(parameter) as the product of the links along it: on a
tiny example, compute dL/dy, dy/dz and dz/dw separately and multiply them. The
chain rule should end up feeling inevitable rather than magical.

For optimisers, start from the problem, not the formula: there is a loss, there
are gradients, the parameters must change, the simplest possible update has
limitations, and the optimiser exists to answer those limitations. Only then
Adam — with the gradient, the running average of the gradient, the running
average of its square, the correction applied to both while those averages are
still young, the step size, and the resulting update, each named separately. Do
not assume the word "moment" carries an obvious meaning.

**For anything about a model, explain in this order.** Skipping a step is what
makes an explanation feel like it started in the middle:

1. The data, and what the model receives.
2. What the model computes from it.
3. What the loss represents.
4. What quantity is being differentiated, and with respect to what.
5. How the gradient is obtained.
6. How the optimiser uses it.
7. The mathematical notation, once the above is in place.
8. The connection to the actual code in this repository.

#### 10.11 Graph networks

Every explanation of a graph network identifies all of the following, in the
terms of the actual problem — for this project, sensors on an engine:

- **The graph** `G = (V, E)`: what one node is, and what an edge means.
- **The node features**: what one row is, and what each column measures.
- **The edges**: why those two nodes should exchange information at all.
- **The neighbourhood**: exactly which nodes may send to which.
- **The message**: what is actually sent.
- **The combination**: how several arriving messages become one thing.
- **The update**: how the node's representation changes as a result.
- **Depth**: one layer reaches roughly one hop, two layers roughly two hops,
  and *why* — a node's new value is built from its neighbours' old values, so
  by the second layer those neighbours' values already carry what their own
  neighbours held.

Never use "message passing" as though it explained itself.

**State the shapes.** `X ∈ R^(N × F)` means N nodes with F features each; then
say what each layer does to that shape:

```
[N, F]  →  graph layer  →  [N, H]
```

Do the same for batches, sequences, adjacency matrices, edge features,
attention coefficients and any single vector standing for a whole graph. If a
tensor has three dimensions, say what all three mean.

**Keep four things apart**, and say where each one's information comes from:
the structure (which nodes are connected), the node features (what was
measured), the learned representations (what the network computed from both),
and any final vector standing for the whole graph.

For attention, answer: what attends to what; what information is compared; how
the compatibility is computed; how the scores are normalised; what the
resulting weights mean; how the weighted values are combined; what is learned;
and what is fixed by the graph. Attention decides the weights **within** a
neighbourhood; it does not create the connections — the adjacency matrix does.

Comparing a graph network with a transformer means comparing how information
can flow, what is built into the architecture before any training, how far one
layer reaches, how position or structure enters, cost, what happens when the
nodes are reordered, and which pairs may attend at all. "Transformers are
global, graph networks are local" is a slogan; say what it means operationally.

#### 10.12 Time, and remaining useful life

For anything spatiotemporal, separate **what changes over time** from **where
the relationships are**, name the three axes — which sensor, which cycle, which
feature — and say what operation runs along each.

For recurrent models, explain why the gates exist before writing the gate
equations, and keep the input, the hidden state, the cell state, the gates and
the output distinct from one another.

For remaining-useful-life work specifically, trace:

```
raw sensor readings → preprocessing → windows over time → node features
        → learned representation → prediction → remaining useful life
```

and be explicit about window length, stride, sequence length, normalisation,
operating conditions, the shape of the degradation, the train/validation/test
split, and above all **leakage** — any path by which information about the test
engines, or from later cycles, could reach a model that is supposed not to have
seen it.

#### 10.13 Explanation methods

Never explain an explanation method by definition alone. Answer seven
questions, every time:

1. **What object is being explained?** A prediction, a node, an edge, a
   subgraph, a feature?
2. **What does the method have access to?** Model internals, gradients,
   weights — or only inputs and outputs?
3. **What does it manipulate?** Features, nodes, edges, inputs, intermediate
   values?
4. **What does it measure?** Contribution, sensitivity, relevance, change in
   the prediction?
5. **What is the resulting explanation?** Per-feature scores, node importance,
   edge importance, a subgraph, a heat map?
6. **What does it assume?**
7. **What does it not prove?**

The last is the one that gets skipped and the one that matters most. A high
SHAP value does not mean the feature caused the prediction in any physical
sense; an attention weight is not importance; a large gradient is not causal
influence. Keep **what the method attributes to a feature**, **a causal effect**
and **real-world importance** apart, and never call an explanation causal
unless causality has actually been established.

**SHAP** is built up, never dropped as a finished formula:

```
prediction → a baseline to compare it against → subsets of features present
        → what one feature adds to a given subset → averaging that over all
        the orders in which the features could have arrived → the Shapley value
```

Use two or three features so every subset can be listed, and say why the order
of arrival matters at all.

**LIME**: original input → perturbed copies of it → the model's predictions on
those copies → a simple model fitted to those points → its coefficients → the
explanation. Explain what "local" means: fitted only near this one input, and
not claimed to hold anywhere else.

**Layer-wise relevance propagation**: separate *activation* from *relevance*.
The question is not which activation was large; it is how the prediction's
total is divided among the things that produced it and handed backwards, share
by share. Show it on a tiny example, and say what stays constant as the shares
move.

**Model-agnostic** methods treat the model as a box mapping input to output, so
they apply to anything but can only learn what varying the input reveals.
**Model-specific** methods can read gradients, weights, intermediate values and
the structure of who sends to whom, so they can measure things the first kind
cannot. Say which is under discussion, and why that access changes what is
measurable.

For graph explanations, say which target is being explained — node features,
node importance, edge importance, a subgraph, or a moment in time — and for
this project's spatiotemporal model answer all four of **what features, where
in the graph, when in time, which relationships**. "Feature importance" is not
the only possible answer and must not be treated as though it were.

#### 10.14 Debugging

Given an error, do not reply with a replacement block of code. Explain, in
order: what the program expected, what it actually received, why those are
incompatible, where exactly the mismatch arises, and why the proposed fix
resolves it. Then give the fix. Where several fixes exist, say which one
preserves the intended design, and why.

#### 10.15 Explaining this codebase

Asked what some code does, do not walk down it line by line unless that was the
request. Start with its purpose, then trace input → transformation →
intermediate data → output, then go into the parts that carry the difficulty.
For an unfamiliar function, give its input, its output, the shapes, its
purpose, anything it changes outside itself, and the machine-learning idea it
implements.

**Never state that this code does something without having looked.** If a
conclusion depends on a file or a function that has not been read, read it. If
it cannot be read, say so plainly — "I cannot establish that from the code I
have looked at" — rather than guessing. This is rules 1 and 2 pointed at the
repository instead of at the paper.

#### 10.16 Research-level precision

Rule 5 divides statements into what the paper says, what we chose and what we
measured. Discussion of the wider literature needs the same discipline: keep
**an empirical result**, **a theoretical property**, **an architectural
intuition** and **speculation** apart, and never present a design choice as a
universal truth.

"This method is better" needs a criterion and evidence. Without them, write
"this may help when …" and say on what basis. Where an alternative would change
the conclusion, say what it is.

#### 10.17 Depth, shape, and the phrases that are banned

Simple questions get simple answers. Conceptual questions get taught properly.
Hard questions about networks, graphs or explanation methods get completeness
in preference to brevity — but structured, with headings, small examples and
diagrams, never an undivided wall of text.

For a basic question — "what is X?" — answer it directly: a plain-language
definition, why X exists, an example, then how it fits the rest of the system.
Do not reach for advanced theory to answer a beginner's question, and do not
imply the answer should already have been known.

For an advanced question, still start at the lowest prerequisite that is
actually needed, and climb: intuition → mechanism → mathematics → example →
code → what it means for this thesis. Skip a layer only when it is genuinely
not needed.

Useful default shapes, to be followed loosely rather than mechanically:

- **A machine-learning question**: what is happening → why it works → step by
  step → tiny example → the mathematics → in the code → the distinction most
  likely to be confused → the key idea in one sentence.
- **A graph-network question**: the problem → the graph → follow one node →
  through one layer → what several layers add → the mathematics → the shapes →
  a tiny graph → why it helps the task.
- **An explanation-method question**: what is being explained → what the method
  can see → what it does → a worked example → what the score means → what it
  does not mean → strengths → limitations → alternatives.

After a difficult section, give one sentence beginning "The key idea is …" that
states the mechanism, not the vocabulary. Where a specific confusion is likely,
plant a one-line checkpoint — "note that the derivative here is with respect to
the parameters, not the label" — but do not turn answers into quizzes.

Do not use advanced notation because it looks precise. Use the simplest
mathematics that carries the idea: a single number first, then vectors, then
matrices. Do not jump from `y = wx + b` to a two-layer network written out in
full unless the extra complexity is actually doing work.

**A metaphor is not an explanation.** "PyTorch has to find the layers", "a
busy node would shout over a quiet one", "put a pulse on A" — each of those was
written here and each had to be replaced by output showing the thing happening.
Show the broken case beside the working one and print both.

**No false simplicity.** "It's very simple", "it's basically just …",
"obviously", "you can think of it as …" and "just imagine …" are banned unless
the sentence that follows genuinely removes complexity instead of hiding it.
Accurate simplicity is the goal; a hard thing made to sound easy is a failure.

**No unexplained borrowed words.** These are permitted, but the first time one
matters it is said in plain terms in the same sentence: *latent
representation*, *feature manifold*, *high-dimensional embedding space*,
*contextual information*, *semantic representation*, *nonlinear
transformation*, *inductive bias*, *receptive field*, *message passing*,
*attention mechanism*, *gradient flow*. `scripts/check_prose.py` enforces a
fixed list of such words across the repository's documents; in conversation the
rule holds anyway, with nothing to enforce it but this paragraph.

#### 10.18 When the explanation did not land

"I don't understand" is evidence that a prerequisite was skipped, not that the
wording was unlucky. Do not restate the same explanation in new phrasing.
Instead: drop the level of abstraction, switch to a concrete example, remove
the jargon, start one conceptual step earlier, name the bridge that was
missing, and build back up from there. If it still does not land, reduce it
again.

#### 10.19 The check before sending

Before finishing any conceptual explanation, ask:

> Could the reader follow every important step from the starting concept to the
> conclusion, without already knowing the answer?

If the answer is no, the explanation is not finished: find the missing bridge
and explain it. A slower answer with one tiny example, one equation explained
carefully and one connection to the code in this repository is worth more than
a fast paragraph containing fifteen technical terms.

---

## Before saying a task is done

```bash
ruff check .          # must be clean
pytest                # 207 tests; must all pass
python scripts/check_prose.py   # borrowed words must be explained where used
```

`check_prose.py` is the floor under rule 9, not the ceiling. It knows a fixed
list of words — *buffer*, *broadcast*, *self-loop*, *ablation* and the rest —
and fails if one is used for the first time in a document with no explanation
beside it. It cannot tell whether a sentence is clear, so a green run means
"none of the known traps", not "this reads well". The check in rule 9 is still
yours to run.

If the change touches `docs/ASSUMPTIONS.md`, a default in `config/`, or
anything the log quotes a number from, also run:

```bash
python scripts/verify_claims.py --fast
```

That recomputes every figure in the log that needs no trained model and fails
if one disagrees with what is written. `--fast` takes seconds; without it, two
more checks train a network and it takes minutes. This exists because the log
has drifted from the code more than once — see the header of that script.

`pytest` needs the real FD001 files in `data/raw/`. Without them the
data-dependent tests skip with a download link rather than failing, leaving
140 passing and 67 skipped. The model, training and explanation tests
additionally need PyTorch; without it they are not collected at all, so 155
of the 207 are collected, 154 pass, and one skips — the figure test that
prints the model's real parameter count.

Both of those counts assume the training runs stored in `reports/`, which are
not committed. A clone that has none also skips the tests about them: with no
data and no stored runs, which is what the CI runner has, it is 134 passing
and 73 skipped.

If a change touches a notebook, re-execute it before committing, so its stored
outputs match its code:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/02_model_fd001.ipynb
```

---

## Environment

- Python 3.10+; the package is importable only after `pip install -e .`
  (the `src/` layout is deliberate — see README section 9).
- PyTorch is optional and only `models/`, `training/` and `explain/` need it.
- All four C-MAPSS sub-datasets are in `data/raw/`, verified against the
  published row and engine counts.
- Long training runs belong in the background; a 5-seed run takes about
  8 minutes on CPU.

## Things that are settled — do not re-litigate

- The 14 sensors and their order come from Table 3 and are load-bearing: the
  adjacency matrix is indexed by position. Import-time assertions guard this.
- The LSTM stage is omitted on purpose (A-11). Implementing it would be
  invention; the paper never states its size, depth or output projection.
- Test labels are capped at 125 (A-02) because the sigmoid output makes higher
  predictions impossible. This can only move reported numbers favourably, and
  that must be stated whenever they are compared with published ones.
- Validation error and test error are not comparable (A-06). Never present one
  as the other, and never read anything into the gap.
