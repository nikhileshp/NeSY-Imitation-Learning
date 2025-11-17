package edu.wisc.cs.will.ILP.Regression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import edu.wisc.cs.will.Utils.Utils;

/**
 * Loads and manages per-grounding weights from fact_weights.txt file.
 * 
 * File format:
 *   predicate(arg1, arg2, ...). weight
 *   
 * Example:
 *   nearby(state1, fish1). 0.98
 *   nearby(state1, fish2). 0.84
 *   oxygen_low(state1). 1.00
 */
public class FactWeightLoader {
	
	// Map: "predicate(arg1,arg2,...)" -> weight
	private Map<String, Double> factWeights;
	private boolean weightsLoaded;
	
	public FactWeightLoader() {
		this.factWeights = new HashMap<String, Double>();
		this.weightsLoaded = false;
	}
	
	/**
	 * Load weights from fact_weights.txt file
	 * 
	 * @param filePath Path to fact_weights.txt
	 * @return true if loaded successfully, false otherwise
	 */
	public boolean loadWeights(String filePath) {
		File file = new File(filePath);
		
		if (!file.exists()) {
			Utils.println("% WARNING: fact_weights.txt not found at: " + filePath);
			Utils.println("% Using default weight of 1.0 for all groundings.");
			return false;
		}
		
		int lineCount = 0;
		int validLines = 0;
		
		try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
			String line;
			
			while ((line = reader.readLine()) != null) {
				lineCount++;
				line = line.trim();
				
				// Skip empty lines and comments
				if (line.isEmpty() || line.startsWith("%")) {
					continue;
				}
				
				// Parse line: "fact. weight"
				if (parseLine(line)) {
					validLines++;
				} else {
					Utils.println("% WARNING: Malformed line " + lineCount + " in fact_weights.txt: " + line);
				}
			}
			
			weightsLoaded = true;
			Utils.println("% Loaded " + validLines + " fact weights from: " + filePath);
			return true;
			
		} catch (IOException e) {
			Utils.println("% ERROR: Failed to load fact_weights.txt: " + e.getMessage());
			return false;
		}
	}
	
	/**
	 * Parse a single line from fact_weights.txt
	 * 
	 * @param line Line in format "fact. weight"
	 * @return true if parsed successfully
	 */
	private boolean parseLine(String line) {
		// Find the period that ends the fact
		int periodIndex = line.indexOf('.');
		if (periodIndex < 0) {
			return false;
		}
		
		// Extract fact (everything before the period)
		String fact = line.substring(0, periodIndex).trim();
		
		// Extract weight (everything after the period)
		String weightStr = line.substring(periodIndex + 1).trim();
		
		if (fact.isEmpty() || weightStr.isEmpty()) {
			return false;
		}
		
		try {
			double weight = Double.parseDouble(weightStr);
			
			// Validate weight
			if (weight < 0) {
				Utils.println("% ERROR: Negative weight not allowed for fact: " + fact);
				return false;
			}
			
			// Normalize fact string (remove extra whitespace)
			String normalizedFact = normalizeFact(fact);
			factWeights.put(normalizedFact, weight);
			return true;
			
		} catch (NumberFormatException e) {
			return false;
		}
	}
	
	/**
	 * Normalize a fact string by removing extra whitespace
	 * 
	 * @param fact Fact string
	 * @return Normalized fact
	 */
	private String normalizeFact(String fact) {
		// Remove spaces around parentheses and commas
		return fact.replaceAll("\\s*\\(\\s*", "(")
		           .replaceAll("\\s*\\)\\s*", ")")
		           .replaceAll("\\s*,\\s*", ",");
	}
	
	/**
	 * Get weight for a specific grounding
	 * 
	 * @param factString Fact as string, e.g., "nearby(state1,fish1)"
	 * @return Weight for this grounding (default 1.0 if not found)
	 */
	public double getWeight(String factString) {
		String normalized = normalizeFact(factString);
		return factWeights.getOrDefault(normalized, 1.0);
	}
	
	/**
	 * Check if weights were successfully loaded
	 * 
	 * @return true if weights loaded
	 */
	public boolean isWeightsLoaded() {
		return weightsLoaded;
	}
	
	/**
	 * Get total number of loaded weights
	 * 
	 * @return Number of fact weights
	 */
	public int getWeightCount() {
		return factWeights.size();
	}
	
	/**
	 * Clear all loaded weights
	 */
	public void clear() {
		factWeights.clear();
		weightsLoaded = false;
	}
}
