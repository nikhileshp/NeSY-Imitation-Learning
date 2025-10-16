# Weighted variance and clause fit in BoostSRL (RDN)

This document explains, concretely and with source references, how BoostSRL computes the “weighted variance” (sum of branch SSEs) that drives clause scoring in relational dependency network (RDN) learning.

Summary
- For each candidate clause at a node, examples that reached the node are partitioned into two branches:
  - TRUE branch: examples covered by the clause’s body
  - FALSE branch: examples not covered at this node
- For each branch separately, the learner fits a constant leaf value λ (lambda) that minimizes the branch’s weighted squared error with respect to the current regression targets (residuals) r_i.
- The branch’s weighted variance is the minimized weighted SSE; the clause’s fit is the sum of the TRUE and FALSE branch SSEs.
- The returned score is −(fit + small penalties). The learner maximizes this returned score, so lower loss (fit) yields a higher (less negative) returned score.

Notation
- r_i: current regression target (residual) for example i
- w_i: example i’s weight
- n_i: number of groundings for example i under the clause (often 1 in the RDN path; can be >1 in other settings)

Branch objective and solution
- For a branch, the model predicts ŷ_i = λ n_i. The branch loss is:
  L(λ) = Σ_i w_i (r_i − λ n_i)^2
- The minimizer (weighted least-squares) is:
  λ* = (Σ_i w_i n_i r_i) / (Σ_i w_i n_i^2)
- The minimized loss (weighted SSE) is:
  SSE_branch = Σ_i w_i r_i^2 − (Σ_i w_i n_i r_i)^2 / (Σ_i w_i n_i^2)

Where this is computed in code

1) Per-branch statistics and formulas (lambda and weighted variance)
```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/Regression/BranchStats.java start=65
public double getLambda(boolean useProbWeights) {
    if (!Double.isNaN(useFixedLambda)) {
        return useFixedLambda;
    }
    if (sumOfNumGroundingSquared == 0) {
        return 0;
    }
    if (sumOfNumGroundingSquaredWithProb == 0) {
        Utils.waitHere("Groundings squared with prob is 0??");
    }
    double lambda =  sumOfOutputAndNumGrounding / sumOfNumGroundingSquared;
    if (useProbWeights) {
        //Utils.waitHere("Computations not correct for vector-based probabilities");
        lambda = sumOfOutputAndNumGrounding / sumOfNumGroundingSquaredWithProb;
    }
    return lambda;
}

public double getWeightedVariance() {
    if (sumOfNumGroundingSquared == 0) {
        return 0;
    }
    return sumOfOutputSquared - (Math.pow(sumOfOutputAndNumGrounding, 2)/sumOfNumGroundingSquared); 
}
```

- Internally, for each example added to a branch, the code accumulates:
  - sumOfOutputSquared        = Σ w_i r_i^2
  - sumOfOutputAndNumGrounding= Σ w_i n_i r_i
  - sumOfNumGroundingSquared  = Σ w_i n_i^2
  (See BranchStats.addNumOutput.)

2) How a clause’s fit is assembled from branches
```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/Regression/RegressionInfoHolderForRDN.java start=29
@Override
public double weightedVarianceAtSuccess() {
    return trueStats.getWeightedVariance();
}

@Override
public double weightedVarianceAtFailure() {
    return falseStats.getWeightedVariance();
}
```

3) The regressionFit used for scoring (sum of TRUE and FALSE branch SSEs), with min-coverage checks
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

    if (!computeWeightedAverage) {
        return getRegressionInfoHolder().variance();
    }

    return getRegressionInfoHolder().weightedVarianceAtSuccess() + getRegressionInfoHolder().weightedVarianceAtFailure();
}
```

4) Returned score (negated loss + small penalties)
```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/ScoreRegressionNode.java start=31
public double scoreThisNode(SearchNode nodeRaw) throws SearchInterrupted {
    SingleClauseNode node  = (SingleClauseNode)nodeRaw;
    double fit     = (forMLNs ? node.regressionFitForMLNs() : node.regressionFit());
    double penalty = scalingPenalties * (getPenalties(node, true, true));
    double score   = fit + penalty;
    if (debugLevel > -1) {
        Utils.println("%     Score = " + Utils.truncate(-score, 6) + " (regressionFit = " + Utils.truncate(fit, 6) + ", penalties=" + penalty + ") for clause:  " + node);
    }
    node.score = -score;
    return -score; // maximize returned score ⇒ minimize (fit + penalties)
}
```

Which examples go to which branch
- Coverage is computed via a Horn clause prover; examples that succeed on the full body (up to this node) form the TRUE branch, and those that fail form the FALSE branch. See SingleClauseNode.computeCoverage() for the proving and “MISSED POS (due to last literal)” messages in logs.

Where n_i (number of groundings) comes from
- For a given example, the number of groundings n_i for the clause body is computed on demand. In the RDN path used in your runs, this is commonly 1, but the code supports counting groundings explicitly:
```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/SingleClauseNode.java start=2042
public long getNumberOfGroundingsForRegressionEx(Example eg) {
    initGroundingCalc();
    LearnOneClause learnClause = ((LearnOneClause)task);
    BindingList theta = learnClause.unifier.unify(this.getClauseHead(), eg.extractLiteral());
    long cached_num = ((RegressionExample)eg).lookupCachedGroundings(this);
    if (cached_num >=0) {
        return cached_num;
    }
    ...
}
```
```java path=/home/nikhilesh/Projects/NeSY-Imitation-Learning/BoostSRL/src/edu/wisc/cs/will/ILP/SingleClauseNode.java start=2145
Set<BindingList> blSet = null;
if (cacheBLs) { blSet = new HashSet<BindingList>();}
num = groundingsCalc.countGroundingsForConjunction(new_body, new ArrayList<Literal>(), blSet);
if (num <= 0) {
    // Utils.waitHere("Number of groundings: " + num + " for " + eg + " in " + this.getClause());
}
```

Interpretation for different clauses
- Clauses that partition examples so that each branch has tighter residuals (r_i) around its own λ (mean) yield smaller TRUE/FALSE branch SSEs and thus smaller fit.
- Example from your run:
  - father(A, B) :- childof(B, A) produced a lower total SSE (fit ≈ 0.83) than father(A, _) :- male(A) (fit ≈ 2.46), so its returned score was higher (−0.83 > −2.46) and it was chosen as best.

Acceptability checks
- Even before comparing fits, a candidate must satisfy per-branch minimum coverage. If either branch has too few examples (by weight), regressionFit returns +Infinity, making the returned score −∞ and the clause is rejected.

Probability-weighted variants
- BranchStats also stores a probability-weighted denominator (sumOfNumGroundingSquaredWithProb) and can compute λ using that when useProbWeights is enabled; the weighted variance used for RDN’s fit in your run uses the standard sums shown above.

How to verify on your machine
- Run with a small number of trees (e.g., -trees 1) and watch for lines like:
  - “Score = -0.833334 (regressionFit = 0.833333, …)”
  - “Score = -2.464287 (regressionFit = 2.464286, …)”
- The less negative score indicates the better (lower-loss) clause.

Pointers
- Branch stats and formulas: BranchStats.java (getLambda, getWeightedVariance)
- Per-branch accumulation: BranchStats.addNumOutput
- Clause fit assembly: RegressionInfoHolderForRDN.weightedVarianceAtSuccess/Failure
- Fit computation and coverage checks: SingleClauseNode.regressionFit
- Final returned score and penalties: ScoreRegressionNode.scoreThisNode
