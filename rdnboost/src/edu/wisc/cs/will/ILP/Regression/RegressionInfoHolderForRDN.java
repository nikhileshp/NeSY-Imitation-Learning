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
import edu.wisc.cs.will.Utils.FactWeights;
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
falseStats.addNumOutput(numGrndg, output, weight, prob.getProbOfBeingTrue());
}

// Optional weighted variant for false branch accumulation.
public void addFailureExampleWeighted(SingleClauseNode caller, Example eg) {
		double baseW = eg.getWeightOnExample();
		double output =  ((RegressionExample) eg).getOutputValue();
		ProbDistribution prob   = ((RegressionRDNExample)eg).getProbOfExample();
		double p = prob.isHasDistribution() ? prob.getProbOfBeingTrue() : 0.5;
		double phi = FactWeights.getInstance().weightForLastLiteral(caller, eg);
		double wEff = baseW * phi;
		double nEff = phi;
		falseStats.addWeighted(nEff, output, wEff, p);
}

	@Override
	public double variance() {
		return (weightedVarianceAtSuccess() + weightedVarianceAtFailure()) / (totalExampleWeightAtSuccess() + totalExampleWeightAtFailure());
	}

	@Override
	public void populateExamples(LearnOneClause task, SingleClauseNode caller) throws SearchInterrupted {
		if (!task.regressionTask) { Utils.error("Should call this when NOT doing regression."); }
		if (caller.getPosCoverage() < 0.0) { caller.computeCoverage(); }
		for (Example posEx : task.getPosExamples()) {
			double weight = posEx.getWeightOnExample();
			double output = ((RegressionExample) posEx).getOutputValue();
			ProbDistribution prob   = ((RegressionRDNExample)posEx).getProbOfExample();
			if (prob.isHasDistribution()) {
				Utils.error("Expected single probability value but contains distribution");
			}
if (!caller.posExampleAlreadyExcluded(posEx)) {
				// Apply optional fact weighting using last-literal grounding.
				double phi = FactWeights.getInstance().weightForLastLiteral(caller, posEx);
				double wEff = weight * phi;
				double nEff = phi;
				trueStats.addWeighted(nEff, output, wEff, prob.getProbOfBeingTrue());
			}
		}
		RegressionInfoHolder totalFalseStats = caller.getTotalFalseBranchHolder() ;
		if (totalFalseStats != null) {
			falseStats = falseStats.add(((RegressionInfoHolderForRDN)totalFalseStats).falseStats);
		}
		// Utils.println("Populated examples: " + trueStats.getNumExamples() + " task: " + caller.getClause());
	}

}
