# BoostSRL — Toy-Father (1 BoostedRDN tree)

This README explains how to reproduce the Toy-Father learning run, how to read the console output, and the math behind learning a single Boosted Relational Dependency Network (BoostedRDN) tree.


## Prerequisites
- Java 8 (OpenJDK 1.8+)
- This repository (BoostSRL) checked out locally


## Command to reproduce
Run from the repository root:

```
bash
java -jar boostsrl-1.1.1.jar -l \
  -train sample/Toy-Father/Father/train \
  -target father \
  -trees 1 \
  -aucJarPath src/edu/wisc/cs/will/DataSetUtils/
```

Outputs appear on stdout and the learned model is saved under:
- sample/Toy-Father/Father/train/models/bRDNs/
- sample/Toy-Father/Father/train/models/WILLtheories/


## What happens during the run (high level)
1) Setup
- Initializes WILL (logic engine), creates dribble logs, sets variable notation, loads background knowledge, modes, libraries.
- Reads facts and examples (train_pos.txt, train_neg.txt, train_bk.txt, train_facts.txt).

2) Prepare ILP search for tree learning
- Creates an outer loop (tree-structured learning) and an inner loop (ILP clause search using best-first search and a scorer).
- Chooses the target predicate father/2 and reports targets/arg types.
- Reports raw vs kept examples (after filtering by arity/target and optional reweighting/bagging).

3) Dataset and gradients
- Builds a regression dataset for boosting. For binary classification with no hidden literals, each example’s gradient is y − p where y∈{0,1} and p is the current predicted probability.
- Prints dataset size, probability computation time, and hidden-literal info (none here).

4) Learn 1 tree
- Starts with the most-general clause (root).
- Expands candidate literals for the root (male/1, childof/2, siblingof/2, …), scores each using a regression fit + penalties.
- Chooses best root literal (childof(B,A)).
- Splits into TRUE/FALSE branches; decides between making a leaf or continuing to expand based on a variance-based stopping rule and minimum positive coverage.
- Expands the TRUE branch once more, best child is male(A). Both TRUE and FALSE branches at this level become leaves due to min coverage.
- Prints the learned tree in decision-tree and clause form; saves the model.

5) Finalization
- Writes Prolog-style helper rules for step length and log prior; prints flattened literals; saves final model files.


## Reading the key print lines
- “Learning 1 trees … for father” — this run is configured to learn a single boosted tree (−trees 1).
- “Dataset size: 19” — total examples used by the current boosting iteration.
- “Variance: X / Set score: 0.0025” — variance at root; the per-branch stopping threshold is set to min(0.0025, 0.25×root-variance).
- “Score = … for clause: father(…) :- …” — best-first search scorer’s report for each candidate clause.
- “Gleaner: New best node found …” — the search monitor tracks the currently best-scoring clause.
- “Creating a TRUE/FALSE-branch leaf because …” — a leaf is created if variance is below the max-acceptable score, or branch’s weighted positive count is too small (coverage constraint), or depth limits would be exceeded.
- “WILL-Produced Tree #1 … FOR father(A,B): if (childof(B,A)) then if (male(A)) …” — the learned decision tree with per-leaf regression output.
- “stepLength_tree1(1.0). logPrior(…) …” — meta-predicates used to compute final scores from all trees plus a log-odds prior.


## The resulting Tree (interpreted)
- Root test: childof(B,A)
  - If TRUE and male(A): return 0.858148935…
  - If TRUE and not male(A): return 0.191482268…
  - If FALSE: return −0.1418510649…

These returned values are regression outputs that, combined with the log prior and the logistic link, yield a probability.


## The model and the math
BoostedRDN fits a stage-wise additive model of relational regression trees for the (log-odds) score of the target predicate instance:

- Score(x) = logPrior + Σ_{t=1..T} stepLength_t · f_t(x)
- Probability P(y=1|x) = sigmoid(Score(x))

For 1 tree (T=1) with default stepLength_1 = 1.0, the model is simply:
- Score(x) = logPrior + f_1(x)
- P = 1 / (1 + exp(−Score(x)))

Leaf value computation
- During learning, each training example i has a gradient g_i = y_i − p_i (binary, no hidden literals). The tree learner partitions examples by relational tests and fits, at each leaf, the mean of these gradients (with appropriate weights). That mean is the leaf’s regression output.

Split criterion and stopping
- The code computes a (weighted) variance of outputs on a branch. A split is favored if it reduces variance. A branch becomes a leaf when:
  - its variance ≤ maxAcceptableNodeScoreToStop (initially chosen as min(0.0025, 0.25×root variance)), or
  - its weighted positive coverage is below 2.1 × minPosCoverage, or
  - tree depth/size limits would be exceeded.

Scoring prints
- The “Score = … (regressionFit = …, penalties= …)” lines come from the scorer combining fit (variance-based) and penalties for complexity; the search maximizes the negative of that combination.

Prior
- logPrior(…) initializes the log-odds before adding any trees. With more trees, the prior is typically the initial bias, while later trees add corrections.


## Files written
- sample/Toy-Father/Father/train/models/bRDNs/father.model(.ckpt): Serialized model (meta + per-tree files)
- sample/Toy-Father/Father/train/models/WILLtheories/father_learnedWILLregressionTrees.txt: Human-readable summary of the learned tree(s)
- sample/Toy-Father/Father/train/models/bRDNs/dotFiles/WILLTreeFor_father0.dot: DOT graph of the tree (if enabled)


## How to interpret/inspect the learned model
- The WILLtheories text contains a tree in both decision-tree and Horn-clause form. You can use it to understand which relational conditions drive the model.
- The Prolog-style rules (stepLength_tree1, logPrior, getScore_…) show how per-tree outputs are combined.


## Appendix: where key prints originate (source pointers)
- RunBoostedRDN.runJob(): “Starting a LEARNING run of bRDN.”
- RunBoostedModels.setupWILLForTrain(): “Calling SETUP.”
- Utils.createDribbleFile(): “Running on host: …”
- HandleFOPCstrings.setVariableIndicator(): “Switching/Unset’ing VarIndicator …”
- LazyHornClausebaseIndexer.resetIndex(): “Resetting the LazyGroundNthArgumentClauseIndex.”
- WILLSetup.createRegressionOuterLooper(): “Calling ILPouterLoop …”, “The outer looper has been created.”
- ILPouterLoop.getInputArgWithDefaultValue(): prints the resolved file args (pos, neg, bk, facts)
- LearnOneClause.readBackgroundTheory(): “Reading background theory …”
- FileParser.loadThisFile(): “Load '../background.txt'.”
- LazyGroundClauseIndex.buildIndexForKey(): “[ LazyGroundClauseIndex ] Building full index …”
- LearnOneClause (ctor): “Read the facts./Have read … facts.”, “LearnOneClause initialized.”
- WILLSetup.setup(): “Have … 'raw' positive examples and kept …”, “processing backup’s for …, POS EX = …, NEG EX = …”
- LearnBoostedRDN.learnNextModel(): “Learn model for: father”
- LearnBoostedRDN.getSampledPosNegEx()/buildDataSet(): “Dataset size: …; Computing probabilities; prob time: …; No hidden examples …”
- ILPouterLoop.resetAll(): “Variance: …; Set score: 0.0025”
- ScoreRegressionNode.scoreThisNode(): “Score = … for clause: father(…) :- …”
- Gleaner.recordNodeBeingScored(): “Gleaner: New best node found …”
- ILPouterLoop (tree code paths): “Expanding node … Will extend: …; Creating a TRUE/FALSE-branch … leaf …; Time for loop …; On cycle # … the best clause found is …”
- TreeStructuredTheory.toPrettyString(): The printed decision tree/clauses and flattened versions
- LearnBoostedRDN.addPrologCodeForUsingAllTrees(): “stepLength_tree1(…); logPrior(…); getScore_…; flattenedLiteralsInThisSetOfTrees(…)”
- ConditionalModelPerPredicate.saveModel(): “Saving model in: …”


## Troubleshooting
- If you see “Unable to access jarfile boostsrl-1.1.1.jar”, ensure you run from the repo root or provide an absolute path to the JAR.
- If BK/Examples aren’t found, make sure the -train directory exists and contains train_pos.txt, train_neg.txt, train_bk.txt, train_facts.txt.
