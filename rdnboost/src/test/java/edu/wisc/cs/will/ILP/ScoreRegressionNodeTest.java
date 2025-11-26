package edu.wisc.cs.will.ILP;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

import edu.wisc.cs.will.FOPC.Literal;
import edu.wisc.cs.will.FOPC.Term;
import edu.wisc.cs.will.FOPC.PredicateName;
import edu.wisc.cs.will.FOPC.HandleFOPCstrings;
import edu.wisc.cs.will.ILP.Regression.FactWeightLoader;

public class ScoreRegressionNodeTest {

    // Stub for FactWeightLoader
    private class StubWeightLoader extends FactWeightLoader {
        // Map: "state,type" -> List<Double>
        private java.util.Map<String, List<Double>> weightsMap = new java.util.HashMap<>();

        public void setWeights(String state, String type, List<Double> weights) {
            weightsMap.put(state + "," + type, weights);
        }

        @Override
        public List<Double> getWeights(String state, String objectType) {
            String key = state + "," + objectType;
            if (weightsMap.containsKey(key)) {
                return weightsMap.get(key);
            }
            return new ArrayList<>();
        }
    }

    // Test subclass to expose protected method and override helper
    private class TestScoreRegressionNode extends ScoreRegressionNode {
        public TestScoreRegressionNode() {
            super();
        }

        @Override
        public long[] computeGroundingCounts(List<Literal> groundedLiterals, List<Literal> originalLiterals, edu.wisc.cs.will.FOPC.BindingList bindings) {
            return super.computeGroundingCounts(groundedLiterals, originalLiterals, bindings);
        }
        
        @Override
        protected String inferObjectTypeFromPredicate(String predicate) {
            if (predicate.contains("p1")) return "type1";
            if (predicate.contains("p2")) return "type2";
            if (predicate.contains("p3")) return "type1"; // p3 also uses type1
            return "testType";
        }
    }

    @Test
    public void testComputeGroundingCounts_ProductLogic() {
        TestScoreRegressionNode node = new TestScoreRegressionNode();
        StubWeightLoader loader = new StubWeightLoader();
        node.setWeightLoader(loader);
        // Threshold 0.5
        node.setGroundingPenaltyParams(0.5, 0.1, 0.5, "min");

        // Setup weights
        // Type1 (for Anon1): {0.2, 0.8} -> 1 high (0.8), 1 low (0.2)
        // Type2 (for Anon2): {0.4, 0.9, 0.6} -> 2 high (0.9, 0.6), 1 low (0.4)
        loader.setWeights("state", "type1", Arrays.asList(0.2, 0.8));
        loader.setWeights("state", "type2", Arrays.asList(0.4, 0.9, 0.6));

        edu.wisc.cs.will.FOPC.HandleFOPCstrings sh = new edu.wisc.cs.will.FOPC.HandleFOPCstrings();
        
        List<Literal> grounded = new ArrayList<>();
        
        // Lit 1: p1(state, anon1) -> infers type1
        Literal l1 = sh.getLiteral(sh.getPredicateName("p1"), Arrays.asList(sh.getStringConstant("state"), sh.getStringConstant("anon1")));
        grounded.add(l1);
        
        // Lit 2: p2(state, anon2) -> infers type2
        Literal l2 = sh.getLiteral(sh.getPredicateName("p2"), Arrays.asList(sh.getStringConstant("state"), sh.getStringConstant("anon2")));
        grounded.add(l2);

        // Expected Logic:
        // Anon1 (Type1) High Count: 1 (0.8)
        // Anon2 (Type2) High Count: 2 (0.9, 0.6)
        // Total High Groundings = 1 * 2 = 2
        
        // Total Groundings = 2 * 3 = 6
        
        long[] counts = node.computeGroundingCounts(grounded, null, null);
        
        org.junit.Assert.assertEquals("High count mismatch", 2, counts[0]);
        org.junit.Assert.assertEquals("Total count mismatch", 6, counts[1]);
    }

    @Test
    public void testComputeGroundingCounts_SharedVariable() {
        TestScoreRegressionNode node = new TestScoreRegressionNode();
        StubWeightLoader loader = new StubWeightLoader();
        node.setWeightLoader(loader);
        node.setGroundingPenaltyParams(0.5, 0.1, 0.5, "min");

        // Anon1 used in two predicates: p1 and p3
        // Both infer "type1"
        // Weights for type1: {0.2, 0.8}
        //
        // Groundings for Anon1:
        // 1. Object 1 (weight 0.2): p1(0.2), p3(0.2) -> min(0.2, 0.2) = 0.2 (Low)
        // 2. Object 2 (weight 0.8): p1(0.8), p3(0.8) -> min(0.8, 0.8) = 0.8 (High)
        
        loader.setWeights("state", "type1", Arrays.asList(0.2, 0.8));
        
        edu.wisc.cs.will.FOPC.HandleFOPCstrings sh = new edu.wisc.cs.will.FOPC.HandleFOPCstrings();
        
        List<Literal> grounded = new ArrayList<>();
        grounded.add(sh.getLiteral(sh.getPredicateName("p1"), Arrays.asList(sh.getStringConstant("state"), sh.getStringConstant("anon1"))));
        grounded.add(sh.getLiteral(sh.getPredicateName("p3"), Arrays.asList(sh.getStringConstant("state"), sh.getStringConstant("anon1"))));

        long[] counts = node.computeGroundingCounts(grounded, null, null);
        
        // Count = 1 (only the 0.8 case is high)
        // Total = 2
        
        org.junit.Assert.assertEquals("High count mismatch", 1, counts[0]);
        org.junit.Assert.assertEquals("Total count mismatch", 2, counts[1]);
    }
}
