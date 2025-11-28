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
			if (debugLevel > 0) {
				Utils.println("%       Penalty breakdown:");
				Utils.println("%         Length/Singleton = " + Utils.truncate(lengthAndSingletonPenalty, 6));
				if (useGroundingPenalty && weightLoader != null) {
					Utils.println("%         Grounding        = " + Utils.truncate(groundingPenalty, 6));
				}
			}
			Utils.println("%       for clause: " + node);
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
		int examplesChecked = 0;
		int examplesWithBindings = 0;
		
		// Enable binding list caching BEFORE computing anything
		// This must be done before computeCoverage() or getNumberOfGroundingsForRegressionEx()
		node.enableBindingListCaching();
		
		// Clear any existing cache to start fresh
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
		examplesChecked++;
		if (node.posExampleAlreadyExcluded(ex)) {
			continue;  // Skip examples that don't satisfy the clause
		}
		
		// Compute groundings for this example to populate cache
		long numGroundings = node.getNumberOfGroundingsForRegressionEx(ex);
		
		// Get cached binding lists for this example
		Set<BindingList> bindings = node.cachedBindingLists.get(ex);
		
		if (bindings == null || bindings.isEmpty()) {
			// Bindings not cached (likely num > 1), skip this example
			continue;
		}
			examplesWithBindings++;
			
	// For each grounding (binding list)
		int groundingDebugCount = 0;
		for (BindingList bl : bindings) {
			// Apply bindings to get grounded literals
			List<Literal> groundedBody = bl.applyTheta(clauseBody);
			
			// Compute aggregated weight for this grounding, passing the binding list
			double aggregatedWeight = computeAggregatedWeight(groundedBody, clauseBody, bl);
			
			// Skip groundings with no multi-argument predicates (indicated by -1)
			if (aggregatedWeight < 0) {
				continue;
			}
			
			// Debug first few groundings
			if (groundingDebugCount < 2 && examplesWithBindings <= 2) {
				Utils.println("%           Grounding " + groundingDebugCount + ": weight=" + aggregatedWeight + " threshold=" + groundingWeightThreshold + " -> " + (aggregatedWeight >= groundingWeightThreshold ? "HIGH" : "LOW"));
				groundingDebugCount++;
			}
			
			// Classify grounding based on threshold
			if (aggregatedWeight >= groundingWeightThreshold) {
				k_high++;
			} else {
				k_low++;
			}
		}
		}
		
	// Also check negative examples
	for (Example ex : theILPtask.getNegExamples()) {
		examplesChecked++;
		if (node.negExampleAlreadyExcluded(ex)) {
			continue;
		}
		
		// Compute groundings for this example to populate cache
		node.getNumberOfGroundingsForRegressionEx(ex);
		
		Set<BindingList> bindings = node.cachedBindingLists.get(ex);
			if (bindings == null || bindings.isEmpty()) {
				continue;
			}
			examplesWithBindings++;
			
			for (BindingList bl : bindings) {
				List<Literal> groundedBody = bl.applyTheta(clauseBody);
				double aggregatedWeight = computeAggregatedWeight(groundedBody, clauseBody, bl);
				
				// Skip groundings with no multi-argument predicates (indicated by -1)
				if (aggregatedWeight < 0) {
					continue;
				}
				
				if (aggregatedWeight >= groundingWeightThreshold) {
					k_high++;
				} else {
					k_low++;
				}
			}
		}
		
		// Compute penalty: negative reward for high attention, positive penalty for low attention
		// (We return positive values to ADD to penalty, which makes score worse)
		// Scale by scalingPenalties to match magnitude of other penalties
		double rawPenalty = -alphaReward * k_high + betaPenalty * k_low;
		double penalty = scalingPenalties * rawPenalty;
		
	// Temporarily always print to debug (debugLevel is static final = 0)
	Utils.println("%       === Grounding Penalty Calculation ===");
	Utils.println("%         Examples checked: " + examplesChecked);
	Utils.println("%         Examples with cached bindings: " + examplesWithBindings);
	Utils.println("%         Total groundings evaluated: " + (k_high + k_low));
	Utils.println("%         High attention groundings (>= " + groundingWeightThreshold + "): " + k_high);
	Utils.println("%         Low attention groundings (< " + groundingWeightThreshold + "): " + k_low);
	Utils.println("%       ");
	Utils.println("%         Raw penalty = -alpha * k_high + beta * k_low");
	Utils.println("%                     = -" + alphaReward + " * " + k_high + " + " + betaPenalty + " * " + k_low);
	Utils.println("%                     = " + (-alphaReward * k_high) + " + " + (betaPenalty * k_low));
	Utils.println("%                     = " + Utils.truncate(rawPenalty, 6));
	Utils.println("%       ");
	Utils.println("%         Scaled penalty = scalingPenalties * rawPenalty");
	Utils.println("%                        = " + scalingPenalties + " * " + Utils.truncate(rawPenalty, 6));
	Utils.println("%                        = " + Utils.truncate(penalty, 6));
		
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
	private double computeAggregatedWeight(List<Literal> groundedLiterals, List<Literal> originalLiterals, BindingList bindings) {
		if (groundedLiterals == null || groundedLiterals.isEmpty()) {
			return 1.0;  // Default: no penalty
		}
		
		// Map: anonVar -> List of (predicate, weights) pairs
		java.util.Map<String, java.util.List<java.util.List<Double>>> anonVarWeightsMap = new java.util.HashMap<>();
		
		// Collect weights for each anonymous variable in each predicate
		for (int i = 0; i < groundedLiterals.size(); i++) {
			Literal groundedLit = groundedLiterals.get(i);
			
			// Skip single-argument predicates (they don't reference game objects)
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
					
					if (objectType != null) {
						// Register this anon variable with its state and type
						weightLoader.registerAnonVariable(anonVar, state, objectType + "0");
						
						// Get cached weights for this anon variable
						java.util.List<Double> anonWeights = weightLoader.getWeightsForAnonVar(anonVar);
						
						if (anonWeights.isEmpty()) {
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
			// No anonymous variables, skip this grounding
			return -1.0;
		}
		
		// Compute Cartesian product of grounding weights
		java.util.List<Double> allGroundingWeights = new java.util.ArrayList<>();
		computeCartesianProduct(anonVarWeightsMap, allGroundingWeights);
		
		if (allGroundingWeights.isEmpty()) {
			return -1.0;
		}
		
		// Apply final aggregation strategy across all grounding combinations
		switch (aggregationStrategy.toLowerCase()) {
			case "min":
				// Return minimum weight across all possible groundings
				double minWeight = Double.MAX_VALUE;
				for (double w : allGroundingWeights) {
					minWeight = Math.min(minWeight, w);
				}
				return minWeight;
				
			case "max":
				// Return maximum weight across all possible groundings
				double maxWeight = 0.0;
				for (double w : allGroundingWeights) {
					maxWeight = Math.max(maxWeight, w);
				}
				return maxWeight;
				
			case "avg":
				// Average weight across all possible groundings
				double sum = 0.0;
				for (double w : allGroundingWeights) {
					sum += w;
				}
				return sum / allGroundingWeights.size();
				
			case "proportion":
				// Proportion of groundings above threshold
				int aboveThreshold = 0;
				for (double w : allGroundingWeights) {
					if (w >= groundingWeightThreshold) {
						aboveThreshold++;
					}
				}
				return (double) aboveThreshold / allGroundingWeights.size();
				
			default:
				Utils.println("% WARNING: Unknown aggregation strategy '" + aggregationStrategy + "', using 'min'");
				double min = Double.MAX_VALUE;
				for (double w : allGroundingWeights) {
					min = Math.min(min, w);
				}
				return min;
		}
	}
	
	/**
	 * Compute Cartesian product of all grounding weights.
	 * For each anonymous variable, we have multiple predicates, each with multiple weights.
	 * We need to compute all combinations where we pick one weight from each predicate,
	 * aggregate them with MIN across predicates (for each anon var),
	 * then combine all anon vars with MIN again.
	 */
	private void computeCartesianProduct(java.util.Map<String, java.util.List<java.util.List<Double>>> anonVarWeightsMap,
	                                      java.util.List<Double> result) {
		// Convert map to list for easier iteration
		java.util.List<String> anonVars = new java.util.ArrayList<>(anonVarWeightsMap.keySet());
		
		// For each anon var, compute Cartesian product of its predicate weights
		java.util.List<java.util.List<Double>> anonVarCombinedWeights = new java.util.ArrayList<>();
		
		for (String anonVar : anonVars) {
			java.util.List<java.util.List<Double>> predicateWeightLists = anonVarWeightsMap.get(anonVar);
			
			// If multiple predicates reference this anon var, compute Cartesian product
			// and take MIN across predicates for each combination
			java.util.List<Double> combinedWeightsForAnon = new java.util.ArrayList<>();
			computePredicateCartesianProduct(predicateWeightLists, 0, new java.util.ArrayList<>(), combinedWeightsForAnon);
			
			anonVarCombinedWeights.add(combinedWeightsForAnon);
		}
		
		// Now compute Cartesian product across all anon vars, taking MIN
		computeFinalCartesianProduct(anonVarCombinedWeights, 0, new java.util.ArrayList<>(), result);
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
	private String inferObjectTypeFromPredicate(String predicate) {
		if (predicate.contains("diver")) return "diver";
		if (predicate.contains("submarine")) return "enemysubmarine";
		if (predicate.contains("enemy")) return "enemy";
		if (predicate.contains("missile")) return "missile";
		if (predicate.contains("oxygen")) return "oxygen";
		return null;
	}
}
