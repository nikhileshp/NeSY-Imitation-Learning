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
import edu.wisc.cs.will.FOPC.Constant;
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
	private static final int MAX_EXAMPLES_FOR_PENALTY = 100;

	private double computeGroundingPenalty(SingleClauseNode node) throws SearchInterrupted {
		LearnOneClause theILPtask = (LearnOneClause) node.task;
		
		// Enable binding list caching BEFORE computing anything
		node.enableBindingListCaching();
		
		// Ensure coverage is computed
		if (node.getPosCoverage() < 0.0) {
			node.computeCoverage();
		}
		
		// Get clause body for grounding
		List<Literal> clauseBody = node.getClauseBody();
		if (clauseBody == null || clauseBody.isEmpty()) {
			return 0.0;  // No body to ground
		}
		
		// --- Positive Examples ---
		List<Example> validPos = new ArrayList<Example>();
		if (theILPtask.getPosExamples() != null) {
			for (Example ex : theILPtask.getPosExamples()) {
				if (!node.posExampleAlreadyExcluded(ex)) {
					validPos.add(ex);
				}
			}
		}
		
		List<Example> subsetPos = validPos;
		if (validPos.size() > MAX_EXAMPLES_FOR_PENALTY) {
			subsetPos = Utils.chooseRandomNfromThisList(MAX_EXAMPLES_FOR_PENALTY, validPos, false);
		}
		
		double k_high_pos = 0;
		double k_low_pos = 0;
		int examplesWithBindingsPos = 0;
		
		for (Example ex : subsetPos) {
			// Compute groundings for this example to populate cache
			long numGroundings = node.getNumberOfGroundingsForRegressionEx(ex);
			
			// Get cached binding lists for this example
			Set<BindingList> bindings = node.cachedBindingLists.get(ex);
			
			if (bindings == null || bindings.isEmpty()) {
				if (node.cachedBindingLists.containsKey(ex)) {
					node.cachedBindingLists.remove(ex);
				}
				continue;
			}
			examplesWithBindingsPos++;
			
			for (BindingList bl : bindings) {
				double aggregatedWeight = computeAggregatedWeight(clauseBody, bl);
				
				if (aggregatedWeight < 0) continue;
				
				if (aggregatedWeight >= groundingWeightThreshold) {
					k_high_pos++;
				} else {
					k_low_pos++;
				}
			}
			node.cachedBindingLists.remove(ex);
		}
		
		// Scale positive counts
		double scalePos = (subsetPos.isEmpty()) ? 0.0 : (double) validPos.size() / subsetPos.size();
		double total_k_high_pos = k_high_pos * scalePos;
		double total_k_low_pos = k_low_pos * scalePos;
		
		
		// --- Negative Examples ---
		List<Example> validNeg = new ArrayList<Example>();
		if (theILPtask.getNegExamples() != null) {
			for (Example ex : theILPtask.getNegExamples()) {
				if (!node.negExampleAlreadyExcluded(ex)) {
					validNeg.add(ex);
				}
			}
		}
		
		List<Example> subsetNeg = validNeg;
		if (validNeg.size() > MAX_EXAMPLES_FOR_PENALTY) {
			subsetNeg = Utils.chooseRandomNfromThisList(MAX_EXAMPLES_FOR_PENALTY, validNeg, false);
		}
		
		double k_high_neg = 0;
		double k_low_neg = 0;
		int examplesWithBindingsNeg = 0;
		
		for (Example ex : subsetNeg) {
			node.getNumberOfGroundingsForRegressionEx(ex);
			
			Set<BindingList> bindings = node.cachedBindingLists.get(ex);
			if (bindings == null || bindings.isEmpty()) {
				if (node.cachedBindingLists.containsKey(ex)) {
					node.cachedBindingLists.remove(ex);
				}
				continue;
			}
			examplesWithBindingsNeg++;
			
			for (BindingList bl : bindings) {
				double aggregatedWeight = computeAggregatedWeight(clauseBody, bl);
				
				if (aggregatedWeight < 0) continue;
				
				if (aggregatedWeight >= groundingWeightThreshold) {
					k_high_neg++;
				} else {
					k_low_neg++;
				}
			}
			node.cachedBindingLists.remove(ex);
		}
		
		// Scale negative counts
		double scaleNeg = (subsetNeg.isEmpty()) ? 0.0 : (double) validNeg.size() / subsetNeg.size();
		double total_k_high_neg = k_high_neg * scaleNeg;
		double total_k_low_neg = k_low_neg * scaleNeg;
		
		
		// --- Combine ---
		double total_k_high = total_k_high_pos + total_k_high_neg;
		double total_k_low = total_k_low_pos + total_k_low_neg;
		
		// Compute penalty
		double rawPenalty = -alphaReward * total_k_high + betaPenalty * total_k_low;
		double penalty = scalingPenalties * rawPenalty;
		
		if (debugLevel > 0) {
			Utils.println("%       === Grounding Penalty Calculation (Subsampled) ===");
			Utils.println("%         Pos Examples: valid=" + validPos.size() + " checked=" + subsetPos.size() + " scale=" + Utils.truncate(scalePos, 2));
			Utils.println("%         Neg Examples: valid=" + validNeg.size() + " checked=" + subsetNeg.size() + " scale=" + Utils.truncate(scaleNeg, 2));
			Utils.println("%         High groundings: " + Utils.truncate(total_k_high, 1) + " (raw pos=" + k_high_pos + ", raw neg=" + k_high_neg + ")");
			Utils.println("%         Low groundings:  " + Utils.truncate(total_k_low, 1) + " (raw pos=" + k_low_pos + ", raw neg=" + k_low_neg + ")");
			Utils.println("%         Scaled penalty = " + Utils.truncate(penalty, 6));
		}
		
		return penalty;
	}
	
	/**
	 * Aggregate weights of multiple grounded predicates using Cartesian product.
	 * For each unique anonymous variable, collect its possible weights.
	 * Then compute all combinations (Cartesian product) and aggregate using min across predicates.
	 * Finally, apply the strategy (min/max) across all combinations.
	 * 
	 * @param originalLiterals List of original literals (before applyTheta)
	 * @param bindings The binding list used for grounding
	 * @return Aggregated weight value
	 */
	private double computeAggregatedWeight(List<Literal> originalLiterals, BindingList bindings) {
		if (originalLiterals == null || originalLiterals.isEmpty()) {
			return 1.0;  // Default: no penalty
		}
		
		// Map: anonVar -> List of (predicate, weights) pairs
		java.util.Map<String, java.util.List<java.util.List<Double>>> anonVarWeightsMap = new java.util.HashMap<>();
		
		// Collect weights for each anonymous variable in each predicate
		for (int i = 0; i < originalLiterals.size(); i++) {
			Literal originalLit = originalLiterals.get(i);
			
			// Skip single-argument predicates (they don't reference game objects)
			if (originalLit.numberArgs() <= 1) {
				continue;
			}
			
			// Typically first arg is state, second is the object (possibly anon)
			if (originalLit.numberArgs() >= 2) {
				Term firstArgOriginal = originalLit.getArgument(0);
				Term secondArgOriginal = originalLit.getArgument(1);
				
				// Resolve terms using bindings
				Term firstArg = resolveTerm(firstArgOriginal, bindings);
				Term secondArg = resolveTerm(secondArgOriginal, bindings);
				
				// Optimization: Avoid toString() and toLowerCase()
				String secondArgName = null;
				if (secondArg instanceof Constant) {
					secondArgName = ((Constant) secondArg).getName();
				} else if (secondArg instanceof Variable) {
					secondArgName = ((Variable) secondArg).getName(); // Use field directly if accessible, or getter
				} else {
					secondArgName = secondArg.toString(); // Fallback
				}

				// Check if second argument is an anonymous variable (starts with "anon" or "Anon" or "_")
				boolean isAnon = false;
				if (secondArgName != null) {
					if (secondArgName.startsWith("anon") || secondArgName.startsWith("Anon") || secondArgName.startsWith("_")) {
						isAnon = true;
					} else if (secondArgName.startsWith("?anon") || secondArgName.startsWith("?_")) {
						isAnon = true;
					}
				}
				
				if (isAnon) {
					String anonVar = secondArgName; // Keep original case
					String predicate = originalLit.predicateName.name; // Keep original case
					
					// Infer object type from predicate name
					String objectType = inferObjectTypeFromPredicate(predicate);
					
					if (objectType != null) {
						// Register this anon variable with its state and type
						String state = firstArg.toString(); 
						
						weightLoader.registerAnonVariable(anonVar, state.toLowerCase(), objectType + "0");
						
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
			return 1.0;
		}
		
		// Compute Cartesian product of grounding weights
		java.util.List<Double> allGroundingWeights = new java.util.ArrayList<>();
		computeCartesianProduct(anonVarWeightsMap, allGroundingWeights);
		
		if (allGroundingWeights.isEmpty()) {
			return 1.0;
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

	private Term resolveTerm(Term term, BindingList bindings) {
		if (term instanceof Variable) {
			Term resolved = bindings.lookup((Variable) term);
			return (resolved != null) ? resolved : term;
		}
		return term;
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
		// Optimization: Check for substrings directly without toLowerCase()
		// We check both lowercase and capitalized versions to be safe, or just use a case-insensitive check logic
		// But simple contains is faster if we know the casing. 
		// Assuming standard casing might be lowercase or CamelCase.
		// Let's check for the specific strings we care about.
		
		if (containsIgnoreCase(predicate, "diver")) return "diver";
		if (containsIgnoreCase(predicate, "submarine")) return "enemysubmarine";
		if (containsIgnoreCase(predicate, "enemy")) return "enemy";
		if (containsIgnoreCase(predicate, "missile")) return "missile";
		if (containsIgnoreCase(predicate, "oxygen")) return "oxygen";
		return null;
	}
	
	private boolean containsIgnoreCase(String src, String what) {
		final int length = what.length();
		if (length == 0)
			return true; // Empty string is contained

		final char firstLo = Character.toLowerCase(what.charAt(0));
		final char firstUp = Character.toUpperCase(what.charAt(0));

		for (int i = src.length() - length; i >= 0; i--) {
			// Quick check before calling regionMatches
			final char ch = src.charAt(i);
			if (ch != firstLo && ch != firstUp)
				continue;

			if (src.regionMatches(true, i, what, 0, length))
				return true;
		}

		return false;
	}
}
