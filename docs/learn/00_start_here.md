# Start here

This is a guided path through a master's thesis. By the end of it you will
know what the software does, why each step is done the way it is, and — just as
importantly — which parts of it are certain and which are judgement calls.

You need no background in machine learning, aircraft engines or graph theory.
Every term is spelled out the first time it appears.

---

## The question, in one paragraph

A jet engine wears out gradually. If an airline can estimate how many more
flights an engine will survive, it can service it before it fails rather than
after. This project estimates that number from sensor readings, using a kind of
neural network that treats the engine's sensors as a **connected network**
rather than a list of unrelated numbers — and then explains each estimate by
saying which sensors drove it.

---

## What this project actually is

It is a **re-implementation study**. One paper was published in 2026 describing
a method called ESTGNN. It came with no source code, and its data availability
statement offers the data only on request. This project rebuilds the method
from the text of the paper alone, runs it on the public NASA benchmark, and
measures the distance between what the paper reports and what the text alone
can produce.

That distance is the result. It is not a failure to reproduce — it is a
measurement of how much a published description leaves unsaid.

---

## The one habit that makes this project readable

Everything written here is one of exactly three kinds of statement, and they
are never allowed to blur into each other.

> **The paper says X.** Verifiable by opening the PDF. Always cited to a
> specific table, equation or section.

> **We chose Y.** The paper did not specify something, so a choice had to be
> made. Every one of these has a numbered entry — A-01, A-02 and so on —
> giving the gap, the choice, the reasoning and how much it matters.

> **We measured Z.** A number this code produced. It comes with the command
> that produced it, and it is an average over at least five random seeds,
> never a single run.

Watch for those three shapes as you read. Almost every disagreement about a
re-implementation turns out to be one of them wearing another's clothes.

There is a fourth label you will meet: a **finding**, numbered F-01 to F-12.
A finding is something learned *about the paper* from the data — for example
that one of the sensors it includes carries almost no information, or that its
own tables contradict each other in a particular place. Findings are not
choices. They are facts about the source, and they are kept separate from our
choices on purpose.

---

## The eight steps of this path

{{figure:pipeline}}

The diagram above is the first half of the work: eight steps that turn three
plain text files from NASA into arrays a network can be trained on. Four more
stages then turn those arrays into a result — build the network, train it,
score it, explain it.

| Page | What you will learn |
|------|---------------------|
| 1. The problem | What remaining useful life is, what a turbofan measures, and what the NASA benchmark contains |
| 2. Graphs | What a graph is, what message passing does, and why three layers is a real limit |
| 3. Explainability | Why a number alone is not enough, and what a Shapley value fairly divides |
| 4. The pipeline | The eight preprocessing steps, and the three that decide whether the experiment is valid at all |
| 5. The model | The network layer by layer, and what its final layer makes impossible |
| 6. Reading results | The four scores, why one seed is not a result, and when a difference is real |
| 7. Evidence | How this project keeps the paper, the choices and the measurements apart |

Each page takes a few minutes. They are meant to be read in order, and each one
assumes only what came before it.

---

## Where to go afterwards

**The two notebooks** run the whole chain on the real NASA files and show every
intermediate number. They are the best next thing after this path — an hour in
notebook 01 makes the source code read like a reference rather than a puzzle.

**The explorers** — the sensor graph, the network, the stored runs, the
assumptions — compute what they show at the moment you open them, from this
repository. Nothing on those pages is a stored copy that could have drifted
from the code.

**The folder READMEs** hold the full argument for each step of the method, one
per folder of the source tree. Reading only those, with no code at all, gives
you the entire case the thesis makes.

---

## A note on honesty

Several claims in this repository were wrong when first written, and were
corrected after being checked. One finding was written from 14 sampled windows
out of 240 and had both its size and its explanation wrong. One assumption
claimed a sensor was unique when seven qualified. One was logged as a choice
when the paper actually states it outright, and had to be withdrawn.

Those corrections are still in the log, marked as corrections rather than
quietly overwritten. That is deliberate. A corrected record is worth more than a
clean one, and you should read this documentation expecting to find the places
where it argues with its own earlier self.
