package edu.wisc.cs.will.ILP;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import edu.wisc.cs.will.DataSetUtils.Example;
import edu.wisc.cs.will.FOPC.BindingList;
import edu.wisc.cs.will.FOPC.Clause;
import edu.wisc.cs.will.FOPC.Literal;
import edu.wisc.cs.will.FOPC.Term;
import edu.wisc.cs.will.FOPC.Variable;
import edu.wisc.cs.will.ILP.Regression.FactWeightLoader;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.stdAIsearch.SearchInterrupted;
import edu.wisc.cs.will.stdAIsearch.SearchNode;

public class ScoreRegressionNode extends ScoreSingleClauseByAccuracy {
	protected final static int debugLevel = 0;   // Used to control output from this project (0 = no output, 1=some, 2=much, 3=all).

	// Note we ADD penalties here, since the final score gets negated.
	private   final static double scalingPenalties = 1; // For regression we might want to shift the penalties since prediction errors might be smaller or larger
	private   final static double bonusForBridgers = 10000.0; // Seems this should suffice, though for some uses of regression it might not.  Don't want to lose the true score, since that'll help sort.
	private boolean forMLNs = false; 			// Score regression node for MLN's
	
	// Distance weights feature (Step 5)
	private FactWeightLoader weightLoader = null;
	private boolean useDistanceWeights = false;
	
	// Grounding-based penalty parameters
	private double groundingWeightThreshold = 0.5;  // Threshold for considering a grounding "attended"
	private double alphaReward = 0.1;  // Reward per high-attention grounding
	private double betaPenalty = 0.5;  // Penalty per low-attention grounding
	private boolean useGroundingPenalty = false;
	private String aggregationStrategy = "min";  // Options: "min", "max", "avg", "proportion"
	
	public ScoreRegressionNode() {
		this(false);
	}
	public ScoreRegressionNode(boolean useMLNs) {
		super();
		forMLNs = useMLNs;
	}
	
	/**
	 * Set the FactWeightLoader for distance-based weighting
	 * @param loader the FactWeightLoader instance
	 */
	public void setWeightLoader(FactWeightLoader loader) {
		this.weightLoader = loader;
		this.useDistanceWeights = (loader != null);
	}
	
	/**
	 * Enable grounding-based penalty with custom parameters
	 * @param threshold Weight threshold for considering a grounding "attended"
	 * @param alpha Reward coefficient for high-attention groundings
	 * @param beta Penalty coefficient for low-attention groundings
	 * @param strategy Aggregation strategy: "min", "max", "avg", "proportion"
	 */
	public void setGroundingPenaltyParams(double threshold, double alpha, double beta, String strategy) {
		this.groundingWeightThreshold = threshold;
		this.alphaReward = alpha;
		this.betaPenalty = beta;
		this.aggregationStrategy = strategy;
		this.useGroundingPenalty = true;
		if (debugLevel > 0) {
			Utils.println("% Grounding penalty enabled: threshold=" + threshold + 
				" alpha=" + alpha + " beta=" + beta + " strategy=" + strategy);
		}
	}
	
	
	public double computeMaxPossibleScore(SearchNode nodeRaw) throws SearchInterrupted {
		SingleClauseNode node = (SingleClauseNode)nodeRaw;
		
		if (debugLevel > 1) { Utils.println("%     computeMaxPossibleScore = " + (-scalingPenalties * getPenalties(node, false, true)) + " for " + node); }
		return -scalingPenalties * getPenalties(node, false, true); // In best case, could end up with NO singleton variables.
	}
	
	public double scoreThisNode(SearchNode nodeRaw) throws SearchInterrupted {
		SingleClauseNode node  = (SingleClauseNode)nodeRaw;
	//	node.computeCoverage(); // Do we need this?
		if (!Double.isNaN(node.score)) { return node.score; }
		double fit     = (forMLNs ? node.regressionFitForMLNs() : node.regressionFit());
		double basePenalty = getPenalties(node, true, true);
		double lengthAndSingletonPenalty = scalingPenalties * basePenalty;
		double groundingPenalty = 0.0;
		
		// Apply grounding-based attention penalty if enabled
		if (useGroundingPenalty && weightLoader != null) {
			groundingPenalty = computeGroundingPenalty(node);
			if (debugLevel > 1) {
				Utils.println("%     Grounding penalty = " + Utils.truncate(groundingPenalty, 6) + " for clause: " + node);
			}
		}
		
	double totalPenalty = lengthAndSingletonPenalty + groundingPenalty;
	
	// Cap total penalty to prevent negative scores
	// Score should always be positive since we're minimizing variance
	if (totalPenalty < -fit * 0.95) {
		totalPenalty = -fit * 0.95; // Allow penalty to reduce score by at most 95%
	}
	
	double score   = fit + totalPenalty; // Add small penalties as a function of length and the number of singleton variables (so shorter better if accuracy the same).
		
		// Enhanced debug output showing penalty breakdown
		if (debugLevel > -1) {
			Utils.println("%     Score = " + Utils.truncate(-score, 6) + " (regressionFit = " + Utils.truncate(fit, 6) + 
				", totalPenalty = " + Utils.truncate(totalPenalty, 6) + ")" );
			
			// Print branch coverage details
			edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder holder = node.getRegressionInfoHolder();
			if (holder != null) {
				double trueCov = holder.totalExampleWeightAtSuccess();
				double falseCov = holder.totalExampleWeightAtFailure();
				
				double truePos = 0;
				double trueNeg = 0;
				if (holder.getTrueStats() != null) {
					truePos = holder.getTrueStats().getNumPositiveOutputs();
					trueNeg = holder.getTrueStats().getNumNegativeOutputs();
				}
				
				double falsePos = 0;
				double falseNeg = 0;
				if (holder.getFalseStats() != null) {
					falsePos = holder.getFalseStats().getNumPositiveOutputs();
					falseNeg = holder.getFalseStats().getNumNegativeOutputs();
				}
				
				Utils.println("%       True Branch:  " + Utils.truncate(trueCov, 2) + " coverage (Pos: " + Utils.truncate(truePos, 2) + ", Neg: " + Utils.truncate(trueNeg, 2) + ")");
				Utils.println("%       False Branch: " + Utils.truncate(falseCov, 2) + " coverage (Pos: " + Utils.truncate(falsePos, 2) + ", Neg: " + Utils.truncate(falseNeg, 2) + ")");
			}
			if (debugLevel >= 0) {
				Utils.println("%       Penalty breakdown:");
				Utils.println("%         Length/Singleton = " + Utils.truncate(lengthAndSingletonPenalty, 6));
				if (useGroundingPenalty && weightLoader != null) {
					Utils.println("%         Grounding        = " + Utils.truncate(groundingPenalty, 6));
				}
			}
			Utils.println("%       for clause: " + node);
			Utils.println("%\n");
		}
		
		//if (node.posCoverage < Double.MIN_VALUE) { return Double.NaN; } // If a node cannot meet the minPosCoverage or theorem proving times out, score as NaN, which will prevent it from being added to OPEN.
		node.score = -score;
		
		// Store penalty information for debug tracking
		node.lastLengthSingletonPenalty = lengthAndSingletonPenalty;
		node.lastGroundingPenalty = groundingPenalty;
		node.lastTotalPenalty = totalPenalty;
		
		if (score < 0) { Utils.error("Should not have a negative score: " + Utils.truncate(-score, 6) + " (regressionFit = " + Utils.truncate(fit, 6) + ", totalPenalty=" + totalPenalty + ") for clause:  " + node); }
		return -score; // Since the code MAXIMIZES, negate here.
	}
	
	public double computeBonusScoreForThisNode(SearchNode nodeRaw) throws SearchInterrupted { // ADD this to the normal score.
		// If a clause ends with a DETERMINATE literal, we want to allow it to be expanded
		// since the determinate literal by itself is (usually) of no help.
		SingleClauseNode node  = (SingleClauseNode)nodeRaw; 
		if (node.endsWithBridgerLiteral()) {
			if (debugLevel > 1) { Utils.waitHere("COMPUTE BRIDGER BONUS (" + Utils.truncate(bonusForBridgers, 3) + "): " + node); }
			return bonusForBridgers; 
		}
		return 0;
	}
	
	/**
	 * Compute penalty based on attention weights of clause groundings.
	 * Returns a value to ADD to penalty (positive = worse, since score is negated).
	 * 
	 * Strategy:
	 * - For each training example, get all groundings of the clause
	 * - For each grounding, aggregate weights of all grounded predicates
	 * - Count k_high (groundings above threshold) and k_low (groundings below threshold)
	 * - Return: -k_high * alpha + k_low * beta (negative because we ADD penalties)
	 */
	private double computeGroundingPenalty(SingleClauseNode node) throws SearchInterrupted {
		LearnOneClause theILPtask = (LearnOneClause) node.task;
		
		int k_high = 0;  // Count of groundings with weight >= threshold
		int k_low = 0;   // Count of groundings with weight < threshold
		
		// Enable binding list caching BEFORE computing anything
		node.enableBindingListCaching();
		node.cachedBindingLists.clear();
		
		// Ensure coverage is computed
		if (node.getPosCoverage() < 0.0) {
			node.computeCoverage();
		}
		
		// Get clause body for grounding
		List<Literal> clauseBody = node.getClauseBody();
		if (clauseBody == null || clauseBody.isEmpty()) {
			return 0.0;  // No body to ground
		}
		
		// Iterate through positive examples
		for (Example ex : theILPtask.getPosExamples()) {
			if (node.posExampleAlreadyExcluded(ex)) {
				continue;  // Skip examples that don't satisfy the clause
			}
			
			// Compute groundings for this example to populate cache
			node.getNumberOfGroundingsForRegressionEx(ex);
			
			// Get cached binding lists for this example
			Set<BindingList> bindings = node.cachedBindingLists.get(ex);
			
			if (bindings == null || bindings.isEmpty()) {
				continue;
			}
			
			for (BindingList bl : bindings) {
				// Apply bindings to get grounded literals
				List<Literal> groundedBody = bl.applyTheta(clauseBody);
				
				// Compute counts for this grounding
				long[] counts = computeGroundingCounts(groundedBody, clauseBody, bl);
				
				long highCount = counts[0];
				long totalCount = counts[1];
				long lowCount = totalCount - highCount;
				
				// Accumulate counts
				k_high += highCount;
				k_low += lowCount;
			}
		}

		// Iterate through negative examples
		for (Example ex : theILPtask.getNegExamples()) {
			if (node.negExampleAlreadyExcluded(ex)) {
				continue;  // Skip examples that don't satisfy the clause
			}
			
			// Compute groundings for this example to populate cache
			node.getNumberOfGroundingsForRegressionEx(ex);
			
			// Get cached binding lists for this example
			Set<BindingList> bindings = node.cachedBindingLists.get(ex);
			
			if (bindings == null || bindings.isEmpty()) {
				continue;
			}
			
			for (BindingList bl : bindings) {
				// Apply bindings to get grounded literals
				List<Literal> groundedBody = bl.applyTheta(clauseBody);
				
				// Compute counts for this grounding
				long[] counts = computeGroundingCounts(groundedBody, clauseBody, bl);
				
				long highCount = counts[0];
				long totalCount = counts[1];
				long lowCount = totalCount - highCount;
				
				// Accumulate counts
				k_high += highCount;
				k_low += lowCount;
			}
		}
		
		// Compute penalty: negative reward for high attention, positive penalty for low attention
		// (We return positive values to ADD to penalty, which makes score worse)
		double rawPenalty = -alphaReward * k_high + betaPenalty * k_low;
		double penalty = scalingPenalties * rawPenalty;
		
		if (debugLevel > 0) {
			Utils.println("% Grounding Penalty: high=" + k_high + ", low=" + k_low + ", penalty=" + penalty);
		}
		
		// CRITICAL: Clear cache and disable caching to prevent memory leak
		node.cachedBindingLists.clear();
		node.disableBindingListCaching();
		
		return penalty;
	}
	
	/**
	 * Aggregate weights of multiple grounded predicates using Cartesian product.
	 * For each unique anonymous variable, collect its possible weights.
	 * Then compute all combinations (Cartesian product) and aggregate using min across predicates.
	 * Finally, apply the strategy (min/max) across all combinations.
	 * 
	 * @param groundedLiterals List of grounded literals in the clause body (after applyTheta)
	 * @param originalLiterals List of original literals (before applyTheta)
	 * @param bindings The binding list used for grounding
	 * @return Aggregated weight value
	 */
	/**
	 * Compute counts of groundings that satisfy the threshold condition.
	 * Uses product-of-counts logic to avoid Cartesian product.
	 * 
	 * @param groundedLiterals List of grounded literals in the clause body
	 * @param originalLiterals List of original literals
	 * @param bindings The binding list used for grounding
	 * @return long array where [0] = high count, [1] = total count
	 */
	protected long[] computeGroundingCounts(List<Literal> groundedLiterals, List<Literal> originalLiterals, BindingList bindings) {
		if (groundedLiterals == null || groundedLiterals.isEmpty()) {
			return new long[]{1, 1}; // Default: 1 grounding, treated as high (no penalty)
		}
		
		// Map: anonVar -> List of (predicate, weights) pairs
		java.util.Map<String, java.util.List<java.util.List<Double>>> anonVarWeightsMap = new java.util.HashMap<>();
		
		// Collect weights for each anonymous variable in each predicate
		for (int i = 0; i < groundedLiterals.size(); i++) {
			Literal groundedLit = groundedLiterals.get(i);
			
			// Skip single-argument predicates
			if (groundedLit.numberArgs() <= 1) {
				continue;
			}
			
			// Typically first arg is state, second is the object (possibly anon)
			if (groundedLit.numberArgs() >= 2) {
				Term firstArg = groundedLit.getArgument(0);
				Term secondArg = groundedLit.getArgument(1);
				
				String state = firstArg.toString().toLowerCase();
				String secondArgStr = secondArg.toString().toLowerCase();
				
				// Check if second argument is an anonymous variable
				if (secondArgStr.startsWith("anon")) {
					String anonVar = secondArgStr;
					String predicate = groundedLit.predicateName.name.toLowerCase();
					
					// Infer object type from predicate name
					String objectType = inferObjectTypeFromPredicate(predicate);
					
					if (objectType != null && weightLoader != null) {
						// Get weights directly using state and object type
						// No need to register anon variable anymore
						java.util.List<Double> anonWeights = weightLoader.getWeights(state, objectType);
						
						if (anonWeights == null || anonWeights.isEmpty()) {
							anonWeights = new java.util.ArrayList<>();
							anonWeights.add(1.0);
						}
						
						// Store these weights for this anon variable
						if (!anonVarWeightsMap.containsKey(anonVar)) {
							anonVarWeightsMap.put(anonVar, new java.util.ArrayList<>());
						}
						anonVarWeightsMap.get(anonVar).add(anonWeights);
					}
				}
			}
		}
		
		if (anonVarWeightsMap.isEmpty()) {
			// No anonymous variables, treat as 1 high grounding
			return new long[]{1, 1};
		}
		
		// Compute counts using product logic
		long totalHighCount = 1;
		long totalCount = 1;
		
		for (String anonVar : anonVarWeightsMap.keySet()) {
			java.util.List<java.util.List<Double>> predicateWeightLists = anonVarWeightsMap.get(anonVar);
			
			// For this anonymous variable, we have multiple lists of weights (one per predicate).
			// We assume these lists are aligned (same objects in same order).
			// We iterate through the indices and check if ALL weights at index i are >= threshold.
			
			int listSize = predicateWeightLists.get(0).size();
			int highCountForVar = 0;
			
			for (int i = 0; i < listSize; i++) {
				boolean allHigh = true;
				for (java.util.List<Double> weights : predicateWeightLists) {
					if (i < weights.size()) {
						if (weights.get(i) < groundingWeightThreshold) {
							allHigh = false;
							break;
						}
					} else {
						// Should not happen if lists are aligned, but handle safely
						allHigh = false; 
						break;
					}
				}
				if (allHigh) {
					highCountForVar++;
				}
			}
			
			totalHighCount *= highCountForVar;
			totalCount *= listSize;
		}
		
		return new long[]{totalHighCount, totalCount};
	}
	
	/**
	 * Recursive helper to compute Cartesian product of predicates for a single anon var.
	 * Takes MIN across predicates for each combination.
	 */
	private void computePredicateCartesianProduct(java.util.List<java.util.List<Double>> predicateWeightLists,
	                                               int predicateIndex,
	                                               java.util.List<Double> current,
	                                               java.util.List<Double> result) {
		if (predicateIndex == predicateWeightLists.size()) {
			// Base case: we have one weight from each predicate, take MIN
			double minWeight = Double.MAX_VALUE;
			for (double w : current) {
				minWeight = Math.min(minWeight, w);
			}
			result.add(minWeight);
			return;
		}
		
		// Recursive case: try each weight from current predicate
		java.util.List<Double> weights = predicateWeightLists.get(predicateIndex);
		for (double weight : weights) {
			current.add(weight);
			computePredicateCartesianProduct(predicateWeightLists, predicateIndex + 1, current, result);
			current.remove(current.size() - 1);
		}
	}
	
	/**
	 * Recursive helper to compute Cartesian product across all anon vars.
	 * Takes MIN across anon vars for each combination.
	 */
	private void computeFinalCartesianProduct(java.util.List<java.util.List<Double>> anonVarCombinedWeights,
	                                           int anonIndex,
	                                           java.util.List<Double> current,
	                                           java.util.List<Double> result) {
		if (anonIndex == anonVarCombinedWeights.size()) {
			// Base case: we have one weight from each anon var, take MIN
			double minWeight = Double.MAX_VALUE;
			for (double w : current) {
				minWeight = Math.min(minWeight, w);
			}
			result.add(minWeight);
			return;
		}
		
		// Recursive case: try each combined weight from current anon var
		java.util.List<Double> weights = anonVarCombinedWeights.get(anonIndex);
		for (double weight : weights) {
			current.add(weight);
			computeFinalCartesianProduct(anonVarCombinedWeights, anonIndex + 1, current, result);
			current.remove(current.size() - 1);
		}
	}
	
	/**
	 * Helper method to infer object type from predicate name
	 */
	protected String inferObjectTypeFromPredicate(String predicate) {
		if (predicate.contains("diver")) return "diver";
		if (predicate.contains("submarine")) return "enemysubmarine";
		if (predicate.contains("enemy")) return "enemy";
		if (predicate.contains("missile")) return "missile";
		if (predicate.contains("oxygen")) return "oxygen";
		return null;
	}
}
