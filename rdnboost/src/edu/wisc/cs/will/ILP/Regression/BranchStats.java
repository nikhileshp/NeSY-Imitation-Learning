package edu.wisc.cs.will.ILP.Regression;

import java.util.ArrayList;
import java.util.List;

import edu.wisc.cs.will.DataSetUtils.Example;
import edu.wisc.cs.will.Utils.Utils;

public class BranchStats {
	// Debug flag controlled by command-line argument -debugScoring
	public static boolean ENABLE_DETAILED_DEBUG = false;
	
	// Track all evaluated clauses for comparison at the end
	public static class ClauseEvaluation {
		public String clause;
		public int trueCount;
		public int falseCount;
		public int truePosGradients;
		public int trueNegGradients;
		public int falsePosGradients;
		public int falseNegGradients;
		public double trueVariance;
		public double falseVariance;
		public double combinedVariance;
		public double score;
		// Variance computation details for TRUE branch
		public double trueSumOutputSquared;
		public double trueSumOutputAndNumGrounding;
		public double trueSumNumGroundingSquared;
		// Variance computation details for FALSE branch
		public double falseSumOutputSquared;
		public double falseSumOutputAndNumGrounding;
		public double falseSumNumGroundingSquared;
		
		public ClauseEvaluation(String clause, int trueCount, int falseCount, 
				int truePosGrad, int trueNegGrad, int falsePosGrad, int falseNegGrad,
				double trueVar, double falseVar, double combined, double score,
				double trueSumOutSq, double trueSumOutAndNum, double trueSumNumSq,
				double falseSumOutSq, double falseSumOutAndNum, double falseSumNumSq) {
			this.clause = clause;
			this.trueCount = trueCount;
			this.falseCount = falseCount;
			this.truePosGradients = truePosGrad;
			this.trueNegGradients = trueNegGrad;
			this.falsePosGradients = falsePosGrad;
			this.falseNegGradients = falseNegGrad;
			this.trueVariance = trueVar;
			this.falseVariance = falseVar;
			this.combinedVariance = combined;
			this.score = score;
			this.trueSumOutputSquared = trueSumOutSq;
			this.trueSumOutputAndNumGrounding = trueSumOutAndNum;
			this.trueSumNumGroundingSquared = trueSumNumSq;
			this.falseSumOutputSquared = falseSumOutSq;
			this.falseSumOutputAndNumGrounding = falseSumOutAndNum;
			this.falseSumNumGroundingSquared = falseSumNumSq;
		}
	}
	
	public static List<ClauseEvaluation> evaluatedClauses = new ArrayList<ClauseEvaluation>();
	
	// Variables for debugging
	private String branchName = "";
	private String currentClause = "";
	private List<Example> examples = new ArrayList<Example>();  // Collect examples for summary
	private List<Double> exampleOutputs = new ArrayList<Double>();  // Track output/gradient values
	
	protected double sumOfOutputSquared = 0;
	//private double sumOfOutput = 0;
	//private double sumOfNumGrounding = 0;
	//private double weightedProb = 0;
	protected double sumOfNumGroundingSquared = 0;
	protected double sumOfNumGroundingSquaredWithProb = 0;
	protected double sumOfOutputAndNumGrounding = 0;
	protected double numExamples 	=	0;
	
	private double useFixedLambda = Double.NaN;
	
	
	// Not used but useful for debugging
	double numNegativeOutputs = 0;
	double numPositiveOutputs = 0;
	
	/**
	 * Set the branch name for debug output
	 */
	public void setBranchName(String name) {
		this.branchName = name;
	}
	
	/**
	 * Set the current clause being evaluated for debug output
	 */
	public void setCurrentClause(String clause) {
		this.currentClause = clause;
	}
	
	/**
	 * Get the current clause for debug output
	 */
	public String getCurrentClause() {
		return this.currentClause;
	}
	
	/**
	 * Add an example to the collected list for debug summary
	 */
	public void addExampleForDebug(Example example, double output) {
		if (ENABLE_DETAILED_DEBUG && example != null) {
			examples.add(example);
			exampleOutputs.add(output);
		}
	}
	
	public void addNumOutput(long num, double output, double weight,double prob) {
		// ===== ACTUAL COMPUTATION (debug output moved to printDebugSummary) =====
		double deno   = prob * (1-prob);
        if (deno < 0.1) {
        	deno = 0.1; 
        }
      //  sumOfNumGrounding += num;
		sumOfNumGroundingSquared += num*num*weight;
      //  sumOfOutput += output;
        sumOfOutputAndNumGrounding += num*output*weight;
        sumOfOutputSquared += output * output*weight;
        if (output > 0 ) {
        	numPositiveOutputs+=weight; 
        } else {
        	numNegativeOutputs+=weight;
        }
        numExamples+=weight;
        sumOfNumGroundingSquaredWithProb += num*num*weight*deno;
	}
	public BranchStats add(BranchStats other) {
		BranchStats newStats = new BranchStats();
		addTo(other, newStats);
		// Copy examples and outputs for debug
		if (ENABLE_DETAILED_DEBUG) {
			newStats.examples.addAll(this.examples);
			newStats.examples.addAll(other.examples);
			newStats.exampleOutputs.addAll(this.exampleOutputs);
			newStats.exampleOutputs.addAll(other.exampleOutputs);
		}
		// Propagate clause and branch name
		newStats.branchName = this.branchName.isEmpty() ? other.branchName : this.branchName;
		newStats.currentClause = this.currentClause.isEmpty() ? other.currentClause : this.currentClause;
		return newStats;
	}
	
	public void addTo(BranchStats other, BranchStats newStats) {
		//newStats.sumOfNumGrounding = this.sumOfNumGrounding + other.sumOfNumGrounding;
		newStats.sumOfNumGroundingSquared = this.sumOfNumGroundingSquared + other.sumOfNumGroundingSquared;
		newStats.sumOfOutputAndNumGrounding = this.sumOfOutputAndNumGrounding + other.sumOfOutputAndNumGrounding;
		//newStats.sumOfOutput = this.sumOfOutput + other.sumOfOutput;
		newStats.sumOfOutputSquared = this.sumOfOutputSquared + other.sumOfOutputSquared;
		newStats.numNegativeOutputs = this.numNegativeOutputs + other.numNegativeOutputs;
		newStats.numPositiveOutputs = this.numPositiveOutputs + other.numPositiveOutputs;
		newStats.numExamples = this.numExamples + other.numExamples;
		newStats.sumOfNumGroundingSquaredWithProb = this.sumOfNumGroundingSquaredWithProb + other.sumOfNumGroundingSquaredWithProb;
		if (!Double.isNaN(this.useFixedLambda) || !Double.isNaN(other.useFixedLambda)) {
			if (this.useFixedLambda != other.useFixedLambda) {
				Utils.waitHere("Different lambdas for " + this.useFixedLambda + " & " + other.useFixedLambda);
			}	else {
				newStats.useFixedLambda = this.useFixedLambda;
			}
		}
	}
	public double getLambda() {
		return getLambda(false);
	}
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
		
		//if (lambda == 0) {
		//	Utils.println(this.toAttrString());
		//}
		return lambda;
	}
	
	public double getWeightedVariance() {
		if (sumOfNumGroundingSquared == 0) {
			return 0;
		}
		double variance = sumOfOutputSquared - (Math.pow(sumOfOutputAndNumGrounding, 2)/sumOfNumGroundingSquared);
		return variance;
	}
	
	/**
	 * Print a summary of examples in this branch for debug (first 10 only)
	 */
	public void printDebugSummary() {
		if (!ENABLE_DETAILED_DEBUG) {
			return;
		}
		
		int totalExamples = examples.size();
		int examplesToShow = Math.min(10, totalExamples);
		
		// Count positive and negative gradients
		int posGradients = 0;
		int negGradients = 0;
		for (Double output : exampleOutputs) {
			if (output > 0) posGradients++;
			else if (output < 0) negGradients++;
		}
		
		Utils.println("\n[" + branchName + " BRANCH]");
		Utils.println("  Total examples: " + totalExamples + " (" + posGradients + " positive gradients, " + 
					  negGradients + " negative gradients)");
		
		if (examplesToShow > 0) {
			Utils.println("  Showing first " + examplesToShow + " example(s) with facts and gradient values:");
			
			for (int i = 0; i < examplesToShow; i++) {
				Example ex = examples.get(i);
				double output = exampleOutputs.get(i);
				String gradientLabel = (output > 0) ? "POS" : ((output < 0) ? "NEG" : "ZERO");
				
				Utils.println("\n    Example " + (i+1) + " [Gradient: " + 
							  String.format("%.4f", output) + " (" + gradientLabel + ")]");
				Utils.println("      " + ex.toString());
			}
			
			if (totalExamples > 10) {
				Utils.println("\n    ... (" + (totalExamples - 10) + " more examples not shown)");
			}
		}
		
		// Calculate and print variance
		if (sumOfNumGroundingSquared > 0) {
			double variance = sumOfOutputSquared - (Math.pow(sumOfOutputAndNumGrounding, 2) / sumOfNumGroundingSquared);
			Utils.println("\n  Variance Calculation:");
			Utils.println("    Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)");
			Utils.println("           = " + sumOfOutputSquared + " - (" + sumOfOutputAndNumGrounding + "^2 / " + sumOfNumGroundingSquared + ")");
			Utils.println("           = " + sumOfOutputSquared + " - " + (Math.pow(sumOfOutputAndNumGrounding, 2) / sumOfNumGroundingSquared));
			Utils.println("    Weighted Variance = " + variance);
		} else {
			Utils.println("\n  Weighted Variance = 0.0 (no examples)");
		}
	}
	
	/**
	 * Get the number of examples in this branch
	 */
	public int getExampleCount() {
		return examples.size();
	}
	
	/**
	 * Get counts of positive and negative gradients
	 */
	public int[] getGradientCounts() {
		int posGradients = 0;
		int negGradients = 0;
		for (Double output : exampleOutputs) {
			if (output > 0) posGradients++;
			else if (output < 0) negGradients++;
		}
		return new int[]{posGradients, negGradients};
	}
	
	/**
	 * Clear the tracked clause evaluations (call at start of tree learning)
	 */
	public static void clearClauseTracking() {
		evaluatedClauses.clear();
	}
	
	/**
	 * Write all evaluated clauses to a file before node selection
	 * @param outputPath Path to write the file (e.g., "rdn_models/seaquest/node_1_true.txt")
	 * @param depth Current tree depth
	 * @param branch Which branch ("true" or "false")
	 */
	public static void writeClausesToFile(String outputPath, int depth, String branch) {
		if (!ENABLE_DETAILED_DEBUG || evaluatedClauses.isEmpty()) {
			return;
		}
		
		try {
			// Ensure directory exists
			java.io.File file = new java.io.File(outputPath);
			file.getParentFile().mkdirs();
			
			java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.FileWriter(file));
			
			// Create separators (Java 8 compatible)
			StringBuilder sep100 = new StringBuilder(100);
			for (int i = 0; i < 100; i++) sep100.append("=");
			String separator100 = sep100.toString();
			
			StringBuilder sep100dash = new StringBuilder(100);
			for (int i = 0; i < 100; i++) sep100dash.append("-");
			String separator100dash = sep100dash.toString();
			
			// Write header
			writer.println(separator100);
			writer.println("CLAUSE EVALUATIONS FOR NODE AT DEPTH " + depth + " (" + branch.toUpperCase() + " BRANCH)");
			writer.println(separator100);
			writer.println();
			writer.println("Total clauses evaluated: " + evaluatedClauses.size());
			writer.println("Sorted by combined variance (ascending - lower is better)");
			writer.println();
			
			// Sort by combined variance (lower is better)
			List<ClauseEvaluation> sorted = new ArrayList<ClauseEvaluation>(evaluatedClauses);
			java.util.Collections.sort(sorted, new java.util.Comparator<ClauseEvaluation>() {
				public int compare(ClauseEvaluation a, ClauseEvaluation b) {
					return Double.compare(a.combinedVariance, b.combinedVariance);
				}
			});
			
			// Write each clause evaluation
			for (int i = 0; i < sorted.size(); i++) {
				ClauseEvaluation eval = sorted.get(i);
				
				writer.println(separator100dash);
				writer.println("RANK " + (i+1) + (i == 0 ? " *** BEST CLAUSE ***" : ""));
				writer.println(separator100dash);
				writer.println();
				writer.println("Clause: " + eval.clause);
				writer.println();
				writer.println("Total examples being split: " + (eval.trueCount + eval.falseCount));
				writer.println();
				writer.println("Split:");
				writer.println("  Examples that SATISFY clause (TRUE branch):  " + eval.trueCount);
				writer.println("    - Positive gradients (from positive training examples): " + eval.truePosGradients);
				writer.println("    - Negative gradients (from negative training examples): " + eval.trueNegGradients);
				writer.println("  Examples that DON'T satisfy clause (FALSE branch): " + eval.falseCount);
				writer.println("    - Positive gradients (from positive training examples): " + eval.falsePosGradients);
				writer.println("    - Negative gradients (from negative training examples): " + eval.falseNegGradients);
				writer.println();
				writer.println("Variance by branch:");
				writer.println();
				writer.println("  TRUE branch variance:  " + String.format("%.6f", eval.trueVariance));
				if (eval.trueSumNumGroundingSquared > 0) {
					writer.println("    Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)");
					writer.println("           = " + String.format("%.6f", eval.trueSumOutputSquared) + 
								   " - (" + String.format("%.6f", eval.trueSumOutputAndNumGrounding) + 
								   "^2 / " + String.format("%.6f", eval.trueSumNumGroundingSquared) + ")");
					double numerator = Math.pow(eval.trueSumOutputAndNumGrounding, 2) / eval.trueSumNumGroundingSquared;
					writer.println("           = " + String.format("%.6f", eval.trueSumOutputSquared) + 
								   " - " + String.format("%.6f", numerator));
					writer.println("           = " + String.format("%.6f", eval.trueVariance));
				}
				writer.println();
				writer.println("  FALSE branch variance: " + String.format("%.6f", eval.falseVariance));
				if (eval.falseSumNumGroundingSquared > 0) {
					writer.println("    Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)");
					writer.println("           = " + String.format("%.6f", eval.falseSumOutputSquared) + 
								   " - (" + String.format("%.6f", eval.falseSumOutputAndNumGrounding) + 
								   "^2 / " + String.format("%.6f", eval.falseSumNumGroundingSquared) + ")");
					double numerator = Math.pow(eval.falseSumOutputAndNumGrounding, 2) / eval.falseSumNumGroundingSquared;
					writer.println("           = " + String.format("%.6f", eval.falseSumOutputSquared) + 
								   " - " + String.format("%.6f", numerator));
					writer.println("           = " + String.format("%.6f", eval.falseVariance));
				}
				writer.println();
				writer.println("Combined Variance: " + String.format("%.6f", eval.combinedVariance));
				writer.println("  Formula: (trueVar + falseVar) / (trueCount + falseCount)");
				writer.println("         = (" + String.format("%.6f", eval.trueVariance) + " + " + 
							   String.format("%.6f", eval.falseVariance) + ") / (" + 
							   eval.trueCount + " + " + eval.falseCount + ")");
				writer.println("         = " + String.format("%.6f", eval.combinedVariance));
				writer.println();
			}
			
			writer.println(separator100);
			writer.println("END OF CLAUSE EVALUATIONS");
			writer.println(separator100);
			
			writer.close();
			Utils.println("\n% Wrote " + evaluatedClauses.size() + " clause evaluations to: " + outputPath);
			
		} catch (java.io.IOException e) {
			Utils.println("% ERROR: Could not write clause evaluations to " + outputPath + ": " + e.getMessage());
		}
	}
	
	/**
	 * Print summary of all evaluated clauses showing which was best
	 */
	public static void printClauseComparison() {
		if (!ENABLE_DETAILED_DEBUG || evaluatedClauses.isEmpty()) {
			return;
		}
		
		// Create separator
		StringBuilder sb = new StringBuilder(100);
		for (int i = 0; i < 100; i++) sb.append("=");
		String separator = sb.toString();
		
		Utils.println("\n\n" + separator);
		Utils.println("CLAUSE COMPARISON SUMMARY - ALL EVALUATED CLAUSES");
		Utils.println(separator);
		Utils.println("\nTotal clauses evaluated: " + evaluatedClauses.size());
		Utils.println("\nAll clauses (sorted by combined variance, best first):");
		Utils.println("");
		
		// Sort by combined variance (lower is better)
		List<ClauseEvaluation> sorted = new ArrayList<ClauseEvaluation>(evaluatedClauses);
		java.util.Collections.sort(sorted, new java.util.Comparator<ClauseEvaluation>() {
			public int compare(ClauseEvaluation a, ClauseEvaluation b) {
				return Double.compare(a.combinedVariance, b.combinedVariance);
			}
		});
		
		// Print header
		Utils.println(String.format("%-4s | %-10s | %-50s | %8s | %8s | %12s", 
				"Rank", "Split", "Clause (truncated)", "TRUE", "FALSE", "Variance"));
		// Create dashes for separator line
		StringBuilder dashes = new StringBuilder(50);
		for (int i = 0; i < 50; i++) dashes.append("-");
		Utils.println(String.format("%-4s-+-%-10s-+-%-50s-+-%-8s-+-%-8s-+-%-12s", 
				"----", "----------", dashes.toString(), "--------", "--------", "------------"));
		
		for (int i = 0; i < sorted.size(); i++) {
			ClauseEvaluation eval = sorted.get(i);
			String truncClause = eval.clause;
			if (truncClause.length() > 50) {
				truncClause = truncClause.substring(0, 47) + "...";
			}
			
			String splitInfo = eval.trueCount + "/" + eval.falseCount;
			String marker = (i == 0) ? " ***" : "";
			
			Utils.println(String.format("%4d | %-10s | %-50s | %8d | %8d | %12.6f%s",
					(i+1), splitInfo, truncClause, eval.trueCount, eval.falseCount, 
					eval.combinedVariance, marker));
		}
		
		Utils.println("");
		Utils.println("*** = Best clause (lowest variance)");
		
		// Show details of best clause
		if (!sorted.isEmpty()) {
			ClauseEvaluation best = sorted.get(0);
			Utils.println("\n" + separator);
			Utils.println("BEST CLAUSE SELECTED");
			Utils.println(separator);
			Utils.println("\nClause: " + best.clause);
			Utils.println("\nSplit:");
			Utils.println("  TRUE branch:  " + best.trueCount + " examples, variance = " + best.trueVariance);
			Utils.println("  FALSE branch: " + best.falseCount + " examples, variance = " + best.falseVariance);
			Utils.println("  Combined variance: " + best.combinedVariance + " (LOWEST - this clause was chosen)");
		}
		
		Utils.println(separator + "\n\n");
	}
	
	public String toAttrString() {
		return 	"% Sum of Output squared		=	" + sumOfOutputSquared + "\n" +
		//"% Sum of Output 				=	" + sumOfOutput + "\n" +
		"% Sum of #groundings squared	=	" + sumOfNumGroundingSquared + "\n" +
		"% Sum of #groundings^2*Probs	=	" + sumOfNumGroundingSquaredWithProb + "\n" +
		//"% Sum of #groundings 			=	" + sumOfNumGrounding + "\n" +
		"% Sum of #groundings*output	=	" + sumOfOutputAndNumGrounding + "\n" +
		"% Num of +ve output			=	" + numPositiveOutputs + "\n" +
		"% Num of -ve output			=	" + numNegativeOutputs ;
	}
	public String toString() {
		return toAttrString() + "\n" + 
				(!Double.isNaN(useFixedLambda) ?
				"% Fixed Lambda					=	" + useFixedLambda + "\n":"") +
				"% Lambda						=	" + getLambda()+ "\n" + 
				"% Prob Lambda					=	" + getLambda(true) ;
	}
	
	public void setZeroLambda() {
		useFixedLambda = 0;
	}
	/**
	 * @return the sumOfOutputSquared
	 */
	public double getSumOfOutputSquared() {
		return sumOfOutputSquared;
	}
	/**
	 * @return the sumOfNumGroundingSquared
	 */
	public double getSumOfNumGroundingSquared() {
		return sumOfNumGroundingSquared; 
	}
	/**
	 * @return the sumOfNumGroundingSquaredWithProb
	 */
	public double getSumOfNumGroundingSquaredWithProb() {
		return sumOfNumGroundingSquaredWithProb;
	}
	/**
	 * @return the sumOfOutputAndNumGrounding
	 */
	public double getSumOfOutputAndNumGrounding() {
		return sumOfOutputAndNumGrounding;
	}
	/**
	 * @return the numExamples
	 */
	public double getNumExamples() {
		return numExamples;
	}
	/**
	 * @return the useFixedLambda
	 */
	public double getUseFixedLambda() {
		return useFixedLambda;
	}
	/**
	 * @return the numNegativeOutputs
	 */
	public double getNumNegativeOutputs() {
		return numNegativeOutputs;
	}
	/**
	 * @return the numPositiveOutputs
	 */
	public double getNumPositiveOutputs() {
		return numPositiveOutputs;
	}
	
	
}

