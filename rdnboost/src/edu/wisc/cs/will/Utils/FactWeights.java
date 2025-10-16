package edu.wisc.cs.will.Utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

import edu.wisc.cs.will.FOPC.Literal;

/**
 * Optional fact-weights provider.
 *
 * Looks for a TSV file mapping grounded facts to weights in [0,1].
 * Default locations (first found wins):
 * - Env FACT_WEIGHTS_FILE
 * - ./fact_weights.tsv (process working directory)
 *
 * If no file is present or a lookup is missing, weight defaults to 1.0.
 */
public class FactWeights {
    private static FactWeights INSTANCE = null;

    private final Map<String, Double> weights = new HashMap<String, Double>(1024);
private boolean initialized = false;
    private String overridePath = null;

    public static FactWeights getInstance() {
        if (INSTANCE == null) { INSTANCE = new FactWeights(); }
        return INSTANCE;
    }

    private FactWeights() {}

    private void initIfNeeded() {
        if (initialized) { return; }
        initialized = true;
String path = (overridePath != null && !overridePath.trim().isEmpty()) ? overridePath : System.getenv("FACT_WEIGHTS_FILE");
        if (path == null || path.trim().isEmpty()) {
            path = new File("fact_weights.tsv").getAbsolutePath();
        }
        File f = new File(path);
        if (!f.exists() || !f.isFile()) { return; }
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) { continue; }
                String[] parts = line.split("\t");
                if (parts.length < 2) { continue; }
                String fact = parts[0].trim();
                double w = 1.0;
                try { w = Double.parseDouble(parts[1].trim()); } catch (Throwable t) { w = 1.0; }
                if (w < 0.0) w = 0.0;
                if (w > 1.0) w = 1.0;
                weights.put(fact, w);
            }
        } catch (Throwable t) {
            // Fail open: keep defaults (weight=1.0)
        }
    }

    /**
     * Allow overriding the weights file at runtime (e.g., from CLI). Passing null clears the override.
     */
    public synchronized void setOverridePath(String path) {
        this.overridePath = path;
        this.initialized = false; // force reload on next lookup
    }

    /**
     * Return a weight in [0,1] for the grounded literal. Defaults to 1.0 if not present.
     */
    public double weightOfLiteral(Literal groundedLiteral) {
        initIfNeeded();
        if (groundedLiteral == null) { return 1.0; }
        Double v = weights.get(groundedLiteral.toString());
        return (v == null ? 1.0 : v.doubleValue());
    }

    /**
     * Convenience: compute weight for last literal of a node grounded with the example's head variables.
     * Returns 1.0 if cannot ground or not found.
     */
    public double weightForLastLiteral(edu.wisc.cs.will.ILP.SingleClauseNode node, edu.wisc.cs.will.DataSetUtils.Example eg) {
        try {
            return node.computeLastLiteralWeightForExample(eg);
        } catch (Throwable t) {
            return 1.0;
        }
    }
}
