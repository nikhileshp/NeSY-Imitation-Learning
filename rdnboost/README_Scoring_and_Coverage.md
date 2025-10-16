# BoostSRL scoring and coverage (RDN learning)

This note explains where the score for regression nodes comes from, how candidate clauses are generated and filtered, and how coverage is computed during learning. It also ties the explanation to the exact source files and the latest run log you asked me to generate.

What was run
- Command:
  java -jar /home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/boostsrl-1.1.1.jar -l -train /home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/sample/Toy-Father/Father/train -target father -trees 1 -aucJarPath /home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/DataSetUtils/ 2>&1 | tee /home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/.boostsrl_run.log
- Fresh log written to: /home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/.boostsrl_run.log

High-level flow
- The learner starts from the root clause (most general): father(_, _).
- It expands the body by adding one literal at a time using the provided modes (male/1, childof/2, siblingof/2, ...), forming candidate SingleClauseNode instances.
- For each candidate, the system:
  1) Computes coverage on positive/negative examples via a Horn clause prover
  2) Computes a regression fit (loss) for the node
  3) Adds small structural penalties (length, singleton vars, unique vars/constants)
  4) Combines into a score used by the search and Gleaner to select the best clause per iteration

Where the regression node score is computed
- Scoring happens in ScoreRegressionNode.scoreThisNode, which wraps SingleClauseNode.regressionFit() and adds penalties. The search maximizes the returned score, so the code returns the negative of a loss-like quantity.

```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/ScoreRegressionNode.java start=31
public double scoreThisNode(SearchNode nodeRaw) throws SearchInterrupted {
	SingleClauseNode node  = (SingleClauseNode)nodeRaw;
//	node.computeCoverage(); // Do we need this?
	if (!Double.isNaN(node.score)) { return node.score; }
	double fit     = (forMLNs ? node.regressionFitForMLNs() : node.regressionFit());
	double penalty = scalingPenalties * (getPenalties(node, true, true)); // + 0.01*node.penaltyForNonDiscrNode());
	
	double score   = fit + penalty; // Add small penalties as a function of length and the number of singleton variables (so shorter better if accuracy the same).
	// Uncomment this for debugging TempEval (TVK)
	//String litString = node.literalAdded.toString();
	// if (debugLevel > -1 || litString.contains("Ve") || litString.contains("Property")) {
	if (debugLevel > -1) {  
		Utils.println("%     Score = " + Utils.truncate(-score, 6) + " (regressionFit = " + Utils.truncate(fit, 6) + ", penalties=" + penalty + ") for clause:  " + node); 
	}
	
	//if (node.posCoverage < Double.MIN_VALUE) { return Double.NaN; } // If a node cannot meet the minPosCoverage or theorem proving times out, score as NaN, which will prevent it from being added to OPEN.
	node.score = -score;
	if (score < 0) { Utils.error("Should not have a negative score: " + Utils.truncate(-score, 6) + " (regressionFit = " + Utils.truncate(fit, 6) + ", penalties=" + penalty + ") for clause:  " + node); }
	return -score; // Since the code MAXIMIZES, negate here.
}
```

- Interpretation:
  - fit is a loss-like value (lower is better), returned by regressionFit().
  - penalty is a small structural cost (see below).
  - score = fit + penalty; the method returns -score so the search maximizes negative loss (i.e., minimizes loss + penalties).
  - The log line “% Score = -X (regressionFit = Y, penalties=Z) …” prints the negated value shown to the search and Gleaner.

What is regressionFit() measuring?
- The fit is computed in SingleClauseNode.regressionFit(). For RDN learning with constants-at-leaves and L2 norm, the code uses the weighted variance of the regression targets on the TRUE and FALSE branches (sum of branch variances). If either branch doesn’t meet the min positive coverage, the fit is set to +Infinity so the candidate is rejected.

```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/SingleClauseNode.java start=1493
public double regressionFit(boolean computeWeightedAverage) throws SearchInterrupted { // This is the expected variance after this node is evaluated (divided by the wgt'ed number of examples if computeWeightedAverage=true).
	LearnOneClause  theILPtask = (LearnOneClause) task;
	
	if (!theILPtask.constantsAtLeaves) { Utils.error("Have not yet implemented constantsAtLeaves = false."); }
	if ( theILPtask.normToUse != 2)    { Utils.error("Have not yet implemented normToUse = " + theILPtask.normToUse + "."); }

	if (getRegressionInfoHolder().totalExampleWeightAtSuccess() < theILPtask.getMinPosCoverage() ||
		getRegressionInfoHolder().totalExampleWeightAtFailure() < theILPtask.getMinPosCoverage()) {
		if (LearnOneClause.debugLevel > 2) {
			Utils.println("regressionFit:\n weightedCountOfExamplesThatSucceed = " + getRegressionInfoHolder().totalExampleWeightAtSuccess() 
									      + "\n weightedCountOfExamplesThatFail    = " + getRegressionInfoHolder().totalExampleWeightAtFailure()
									      + "\n getMinPosCoverage                  = " + theILPtask.getMinPosCoverage());
		}
		return Double.POSITIVE_INFINITY;  // Bad clauses get posCoverage=0 and we don't want to keep such clauses.  Remember we NEGATE the score, so a high score here is bad.
	}
		
	// TODO(test)
	if (!computeWeightedAverage) {
		return getRegressionInfoHolder().variance();
	}
	
	return getRegressionInfoHolder().weightedVarianceAtSuccess() + getRegressionInfoHolder().weightedVarianceAtFailure();
}
```

- Practically, lower weighted variance = better fit; after negation, the best candidates have scores closest to zero (e.g., -0.83 is better than -2.46).

What are the penalties?
- Small tie-breaking regularizers applied to discourage overly complex clauses and repeated patterns, implemented by ScoreSingleClauseByAccuracy.getPenalties().

```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/ScoreSingleClauseByAccuracy.java start=39
protected double getPenalties(SingleClauseNode node, boolean includeSingletonCount, boolean includeRepeatedPredicates) {
	List<Variable> allVariables = node.collectAllVariables();
	List<Constant> allConstants = node.collectAllConstants();
	if (includeRepeatedPredicates) { pNamesSeen.clear(); }
	double bodyCost        =                              node.getCost();
	double singletonVars   = (includeSingletonCount     ? node.countOfSingletonVars(allVariables)      : 0.0);
	double repeatedLits    = (includeRepeatedPredicates ? node.discountForRepeatedLiterals(pNamesSeen) : 0.0);
	double uniqueVars      =                              node.countOfUniqueVars(     allVariables);
	double uniqueConstants =                              node.countOfUniqueConstants(allConstants);
	...
	return                              multiplerForBodyCost         * bodyCost 
		 + (includeSingletonCount     ? multiplierForSingletonVars   * singletonVars : 0.0)
		 - (includeRepeatedPredicates ? multiplerForBodyCost         * repeatedLits  : 0.0)
		 +                              multiplierForUniqueVars      * uniqueVars
		 +                              multiplierForUniqueConstants * uniqueConstants;
}
```

How coverage is computed
- Coverage is calculated by proving each example against the current clause using the HornClauseProver, counting weighted positives (posCoverage) and negatives (negCoverage). The learner also prunes early if min positive coverage cannot be reached.

```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/SingleClauseNode.java start=739
public void computeCoverage() throws SearchInterrupted {
	LearnOneClause   theILPtask = (LearnOneClause) task;
	HornClauseProver prover     = theILPtask.getProver();
	Literal          target     = getTarget();
	List<Literal>    clauseBody = getClauseBody();  // To save space in OPEN, compute this when needed.
	SingleClauseNode parent     = getParentNode();
	boolean          firstTime            = false;
	boolean          tookTooLong          = false;
	long             totalResolutions     = 0;
	boolean          stopWhenUnacceptable = theILPtask.stopWhenUnacceptableCoverage; // Don't continue to prove examples when impossible to meet the minPosCoverage and minPrecision specifications.

	int localDebugLevel = Math.max(-2, LearnOneClause.debugLevel); // Change this line to get more info on false pos/negs.
	List<List<Literal>> optimizedClauseBodies = null;

	// To save time, if posCoverage is not going to reach theILPtask.minPosCoverage stop.
	if (getPosCoverage() < 0.0) {
		extraString = null; // Reset this whenever the coverage changes.
		if (localDebugLevel > 1) { Utils.println("%     computeCoverage: clauseBody = " + clauseBody); }
		double maxPossiblePosCoverage = 0.0;
		int    counter                = 0;
		int    numberPos              = Utils.getSizeSafely(theILPtask.getPosExamples());
		int    numberPosPossible      = 0;
		if (stopWhenUnacceptable) for (Example posEx : theILPtask.getPosExamples()) if (parent == null || !parent.posExampleAlreadyExcluded(posEx)) { // Don't look at THIS node or we'll have an infinite loop.
			maxPossiblePosCoverage += posEx.getWeightOnExample(); // See how much is possible
			numberPosPossible++;
		}
		setPosCoverage(0.0);
		firstTime = true;
		...
}
```

- The log’s “MISSED POS (due to last literal)” lines come from computeCoverage when an example fails at the last added body literal.

How clauses are accepted/rejected and the “best” is chosen
- The Gleaner is responsible for collecting candidates, enforcing acceptability constraints (min positive coverage, max negative coverage, min precision, not the root, etc.), and tracking the best clause by score.

```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/Gleaner.java start=259
// Keep track of the best clause overall, assuming it meets the acceptability criteria.
if (acceptable) {
	if (LearnOneClause.debugLevel > 2) { Utils.println("% Acceptable (score = " + Utils.truncate(score, 4) + "): " + clause ); }
	nodeCounterAcceptable++;
	if (score > bestScore) {
		bestScore = score;
		bestNode  = clause;
		changedAtThisNode = nodeCounterAll;
		if (LearnOneClause.debugLevel > -1) { Utils.println("% Gleaner: New best node found (score = " + Utils.truncate(score, 6) + "): " + nodeBeingCreated); }
	} else if (LearnOneClause.debugLevel > 1) {
		Utils.println("Acceptable but did not beat the score of: " + Utils.truncate(bestScore, 4));
	}
	
}
```

- Because the scoring function returns the negative of a loss, “better” means less negative (e.g., -0.83 beats -2.46).

Example from the latest run (Toy-Father)
- The learner tried several one-literal clauses. Three that became acceptable:
  - father(A, _) :- male(A).  Score = -2.464287; covers 7/11 positives
  - father(_, A) :- male(A).  Score = -Infinity when min branch coverage fails (still used for gleaner binning)
  - father(A, B) :- childof(B, A).  Score = -0.833334; covers 6/11 positives
- The best node (highest score) was father(A, B) :- childof(B, A), which then became the split at the learned tree’s root with leaf outputs:
  - If childof(B,A): output = 0.6914822684
  - Else: output = -0.1418510649
- You can see these values emitted in the log’s “WILL-Produced Tree #1 …” and the clause forms that follow.

Notes
- You may see a benign FileNotFoundException early for train\train_bk.txt on Linux. This is due to a Windows-style path in the discretization checker; the run continues with the correct absolute paths afterwards.
- Acceptability thresholds like min positive coverage and min precision are set within the learning task and enforced by Gleaner before a candidate can become “best.”

Pointers
- Scoring driver: ScoreRegressionNode.scoreThisNode
- Fit: SingleClauseNode.regressionFit
- Penalties: ScoreSingleClauseByAccuracy.getPenalties
- Coverage: SingleClauseNode.computeCoverage
- Selection/acceptability: Gleaner.addToGleaner
