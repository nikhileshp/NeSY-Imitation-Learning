/**
 * 
 */
package edu.wisc.cs.will.ILP.Regression;

import edu.wisc.cs.will.Boosting.RDN.RegressionRDNExample;
import edu.wisc.cs.will.DataSetUtils.Example;
import edu.wisc.cs.will.DataSetUtils.RegressionExample;
import edu.wisc.cs.will.ILP.LearnOneClause;
import edu.wisc.cs.will.ILP.SingleClauseNode;
import edu.wisc.cs.will.Utils.ProbDistribution;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.stdAIsearch.SearchInterrupted;

/**
 * @author tkhot
 *
 */
public class RegressionInfoHolderForRDN extends RegressionInfoHolder {
	
	public RegressionInfoHolderForRDN() {
		trueStats = new BranchStats();
		falseStats = new BranchStats();
		// Set branch names for debug output
		trueStats.setBranchName("TRUE");
		falseStats.setBranchName("FALSE");
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#weightedVarianceAtSuccess()
	 */
	@Override
	public double weightedVarianceAtSuccess() {		
		return trueStats.getWeightedVariance();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#weightedVarianceAtFailure()
	 */
	@Override
	public double weightedVarianceAtFailure() {
		return falseStats.getWeightedVariance();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#totalExampleWeightAtSuccess()
	 */
	@Override
	public double totalExampleWeightAtSuccess() {
		return trueStats.getNumExamples();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#totalExampleWeightAtFailure()
	 */
	@Override
	public double totalExampleWeightAtFailure() {
		return falseStats.getNumExamples();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#meanAtSuccess()
	 */
	@Override
	public double meanAtSuccess() {
		return trueStats.getLambda();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#meanAtFailure()
	 */
	@Override
	public double meanAtFailure() {
		return falseStats.getLambda();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#addFailureStats(edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder)
	 */
	@Override
	public RegressionInfoHolder addFailureStats(RegressionInfoHolder addThis) {
		RegressionInfoHolderForRDN regHolder = new RegressionInfoHolderForRDN();
		if (addThis == null) {
			regHolder.falseStats = this.falseStats.add(new BranchStats());
		} else {
			regHolder.falseStats = this.falseStats.add(((RegressionInfoHolderForRDN)addThis).falseStats);
		}
		return regHolder;
	}


	@Override
	public void addFailureExample(Example eg, long numGrndg, double weight) {
		double output =  ((RegressionExample) eg).getOutputValue();
		ProbDistribution prob   = ((RegressionRDNExample)eg).getProbOfExample();
		if (prob.isHasDistribution()) {
			Utils.error("Expected single probability value but contains distribution");
		}
		// Add example to debug collection with its gradient/output value
		falseStats.addExampleForDebug(eg, output);
		falseStats.addNumOutput(numGrndg, output, weight, prob.getProbOfBeingTrue());
	}

	@Override
	public double variance() {
		return (weightedVarianceAtSuccess() + weightedVarianceAtFailure()) / (totalExampleWeightAtSuccess() + totalExampleWeightAtFailure());
	}

	@Override
	public void populateExamples(LearnOneClause task, SingleClauseNode caller) throws SearchInterrupted {
		if (!task.regressionTask) { Utils.error("Should call this when NOT doing regression."); }
		if (caller.getPosCoverage() < 0.0) { caller.computeCoverage(); }
		
		// Set the current clause on both branches for debug output
		String clauseStr = (caller.getClause() != null) ? caller.getClause().toString() : "";
		trueStats.setCurrentClause(clauseStr);
		falseStats.setCurrentClause(clauseStr);
		
		for (Example posEx : task.getPosExamples()) {
			double weight = posEx.getWeightOnExample();
			double output = ((RegressionExample) posEx).getOutputValue();
			ProbDistribution prob   = ((RegressionRDNExample)posEx).getProbOfExample();
			if (prob.isHasDistribution()) {
				Utils.error("Expected single probability value but contains distribution");
			}
			if (!caller.posExampleAlreadyExcluded(posEx)) {
				// Add example to debug collection with its gradient/output value
				trueStats.addExampleForDebug(posEx, output);
				trueStats.addNumOutput(1, output, weight, prob.getProbOfBeingTrue());		
			}
		}
		RegressionInfoHolder totalFalseStats = caller.getTotalFalseBranchHolder() ;
		if (totalFalseStats != null) {
			// Merge false stats while preserving branch name and clause
			BranchStats mergedFalse = falseStats.add(((RegressionInfoHolderForRDN)totalFalseStats).falseStats);
			mergedFalse.setBranchName("FALSE");
			mergedFalse.setCurrentClause(clauseStr);
			falseStats = mergedFalse;
		}
		
		// Print debug summary after both branches are populated
		printClauseEvaluationSummary();
	}
	
	/**
	 * Print a complete summary of clause evaluation showing both branches
	 */
	private void printClauseEvaluationSummary() {
		if (!BranchStats.ENABLE_DETAILED_DEBUG) {
			return;
		}
		
		// Create separator
		StringBuilder sb = new StringBuilder(90);
		for (int i = 0; i < 90; i++) sb.append("=");
		String separator = sb.toString();
		
		Utils.println("\n" + separator);
		Utils.println("CLAUSE EVALUATION");
		Utils.println(separator);
		
		// Print the clause
		String clauseStr = trueStats.getCurrentClause();
		if (clauseStr != null && !clauseStr.isEmpty()) {
			Utils.println("\nClause: " + clauseStr);
		}
		
		// Print counts
		int trueCount = trueStats.getExampleCount();
		int falseCount = falseStats.getExampleCount();
		Utils.println("\nSplit: " + trueCount + " examples satisfy clause (TRUE), " + 
					  falseCount + " do not (FALSE)");
		
		// Print TRUE branch summary (first 10 examples)
		trueStats.printDebugSummary();
		
		// Print FALSE branch summary (first 10 examples)
		falseStats.printDebugSummary();
		
		// Calculate variances
		double trueVar = weightedVarianceAtSuccess();
		double falseVar = weightedVarianceAtFailure();
		double trueWeight = totalExampleWeightAtSuccess();
		double falseWeight = totalExampleWeightAtFailure();
		double totalWeight = trueWeight + falseWeight;
		double combinedVariance = variance();
		
		// Print combined score
		Utils.println("\n" + separator);
		Utils.println("SPLIT SCORE (Combined Variance)");
		Utils.println(separator);
		Utils.println("  Formula: (trueVar + falseVar) / (trueWeight + falseWeight)");
		Utils.println("         = (" + trueVar + " + " + falseVar + ") / (" + trueWeight + " + " + falseWeight + ")");
		Utils.println("         = " + (trueVar + falseVar) + " / " + totalWeight);
		Utils.println("\n  Combined Variance = " + combinedVariance);
		Utils.println("  (Lower is better - algorithm seeks to MINIMIZE variance)");
		Utils.println(separator + "\n");
		
		// Get gradient counts for both branches
		int[] trueGradCounts = trueStats.getGradientCounts();
		int[] falseGradCounts = falseStats.getGradientCounts();
		
		// Record this evaluation for later comparison with all details
		BranchStats.ClauseEvaluation eval = new BranchStats.ClauseEvaluation(
			clauseStr, trueCount, falseCount,
			trueGradCounts[0], trueGradCounts[1],  // true pos/neg gradients
			falseGradCounts[0], falseGradCounts[1], // false pos/neg gradients
			trueVar, falseVar, combinedVariance, Double.NaN,
			trueStats.getSumOfOutputSquared(), 
			trueStats.getSumOfOutputAndNumGrounding(),
			trueStats.getSumOfNumGroundingSquared(),
			falseStats.getSumOfOutputSquared(),
			falseStats.getSumOfOutputAndNumGrounding(),
			falseStats.getSumOfNumGroundingSquared());
		BranchStats.evaluatedClauses.add(eval);
	}

}
