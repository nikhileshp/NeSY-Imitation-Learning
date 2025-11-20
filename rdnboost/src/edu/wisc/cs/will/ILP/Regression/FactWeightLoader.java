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
	
	// Cache for anonymous variable resolution
	// Maps anon variable (e.g., "anon456") to (state, objectType) pair
	private Map<String, StateObjectPair> anonVarCache;
	// Maps (state, objectType) to list of all weights for that object across predicates
	private Map<StateObjectPair, java.util.List<Double>> stateObjectWeights;
	
	// Helper class to store (state, objectType) pairs
	private static class StateObjectPair {
		public final String state;
		public final String objectType;
		
		public StateObjectPair(String state, String objectType) {
			this.state = state;
			this.objectType = objectType;
		}
		
		@Override
		public boolean equals(Object o) {
			if (this == o) return true;
			if (!(o instanceof StateObjectPair)) return false;
			StateObjectPair that = (StateObjectPair) o;
			return state.equals(that.state) && objectType.equals(that.objectType);
		}
		
		@Override
		public int hashCode() {
			return state.hashCode() * 31 + objectType.hashCode();
		}
		
		@Override
		public String toString() {
			return state + "," + objectType;
		}
	}
	
	public FactWeightLoader() {
		this.factWeights = new HashMap<String, Double>();
		this.weightsLoaded = false;
		this.anonVarCache = new HashMap<String, StateObjectPair>();
		this.stateObjectWeights = new HashMap<StateObjectPair, java.util.List<Double>>();
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
	 * Get all matching fact weights where anonymous variables (anon*) are treated as wildcards.
	 * For example, "visiblediver(srz123,anon456)" will match all facts like:
	 * - visiblediver(srz123,diver0)
	 * - visiblediver(srz123,diver1)
	 * 
	 * @param factPattern Fact pattern with possible anon* variables
	 * @return List of weights for all matching facts (empty if none found)
	 */
	public java.util.List<Double> getMatchingWeights(String factPattern) {
		java.util.List<Double> matches = new java.util.ArrayList<Double>();
		
		// Normalize the pattern
		String normalized = normalizeFact(factPattern);
		
		// Check if pattern contains anonymous variables
		if (!normalized.contains("anon")) {
			// No wildcards, just do direct lookup
			Double weight = factWeights.get(normalized);
			if (weight != null) {
				matches.add(weight);
			}
			return matches;
		}
		
		// Extract predicate name and arguments
		int openParen = normalized.indexOf('(');
		int closeParen = normalized.lastIndexOf(')');
		if (openParen < 0 || closeParen < 0) {
			return matches;
		}
		
		String predName = normalized.substring(0, openParen);
		String argsStr = normalized.substring(openParen + 1, closeParen);
		String[] patternArgs = argsStr.split(",");
		
		// Find all facts with same predicate name and matching non-anon arguments
		for (Map.Entry<String, Double> entry : factWeights.entrySet()) {
			String factKey = entry.getKey();
			
			// Check if same predicate
			if (!factKey.startsWith(predName + "(")) {
				continue;
			}
			
			// Extract arguments from fact
			int factOpenParen = factKey.indexOf('(');
			int factCloseParen = factKey.lastIndexOf(')');
			if (factOpenParen < 0 || factCloseParen < 0) {
				continue;
			}
			
			String factArgsStr = factKey.substring(factOpenParen + 1, factCloseParen);
			String[] factArgs = factArgsStr.split(",");
			
			// Check if argument counts match
			if (factArgs.length != patternArgs.length) {
				continue;
			}
			
			// Check if all non-anon arguments match
			boolean allMatch = true;
			for (int i = 0; i < patternArgs.length; i++) {
				String patternArg = patternArgs[i].trim();
				String factArg = factArgs[i].trim();
				
				// If pattern has anon*, it's a wildcard - any value matches
				if (patternArg.startsWith("anon")) {
					continue;
				}
				
				// Otherwise, must match exactly
				if (!patternArg.equals(factArg)) {
					allMatch = false;
					break;
				}
			}
			
			if (allMatch) {
				matches.add(entry.getValue());
			}
		}
		
		return matches;
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
	 * Register an anonymous variable with its state and object type.
	 * This builds the cache for efficient weight lookup.
	 * 
	 * @param anonVar Anonymous variable name (e.g., "anon456")
	 * @param state State ID (e.g., "srz123")
	 * @param objectName Object name from fact (e.g., "diver0", "enemy1")
	 */
	public void registerAnonVariable(String anonVar, String state, String objectName) {
		// Extract object type from object name
		String objectType = extractObjectType(objectName);
		
		if (objectType != null) {
			StateObjectPair pair = new StateObjectPair(state, objectType);
			anonVarCache.put(anonVar, pair);
			
			// Build the weight list for this (state, objectType) if not already done
			if (!stateObjectWeights.containsKey(pair)) {
				buildWeightListForStateObject(state, objectType);
			}
		}
	}
	
	/**
	 * Extract object type from object name.
	 * Examples: "diver0" -> "diver", "enemy1" -> "enemy", "enemysubmarine0" -> "enemysubmarine"
	 */
	private String extractObjectType(String objectName) {
		if (objectName == null || objectName.isEmpty()) {
			return null;
		}
		
		// Remove trailing digits to get object type
		// Handle cases like "diver0", "enemy12", "enemysubmarine0"
		int i = objectName.length() - 1;
		while (i >= 0 && Character.isDigit(objectName.charAt(i))) {
			i--;
		}
		
		if (i < 0) {
			return null; // All digits, not a valid object name
		}
		
		return objectName.substring(0, i + 1);
	}
	
	/**
	 * Build a list of all weights for a given (state, objectType) pair.
	 * This looks up all facts involving this object and collects their weights,
	 * excluding predicates that always have weight 1.0 (like visible* predicates).
	 */
	private void buildWeightListForStateObject(String state, String objectType) {
		java.util.List<Double> weights = new java.util.ArrayList<Double>();
		
		// Predicates to check for this object type
		// Skip "visible*" predicates as they're always 1.0
		String[] relevantPredicates = getRelevantPredicates(objectType);
		
		// Look for facts with pattern: predicate(state, objectType<digit>)
		for (String predicate : relevantPredicates) {
			// Try different object indices (e.g., diver0, diver1, ..., diver9)
			for (int i = 0; i < 10; i++) {
				String objectName = objectType + i;
				String factKey = predicate + "(" + state + "," + objectName + ")";
				
				Double weight = factWeights.get(factKey);
				if (weight != null) {
					weights.add(weight);
				}
			}
		}
		
		StateObjectPair pair = new StateObjectPair(state, objectType);
		stateObjectWeights.put(pair, weights);
	}
	
	/**
	 * Get relevant predicates for a given object type (excluding always-1.0 predicates).
	 */
	private String[] getRelevantPredicates(String objectType) {
		switch (objectType) {
			case "diver":
				return new String[]{"leftofdiver", "rightofdiver", "aboveofdiver", "belowofdiver", "nearbydiver"};
			case "enemy":
				return new String[]{"leftofenemy", "rightofenemy", "aboveofenemy", "belowofenemy", "nearbyenemy", "samelevelasenemy"};
			case "enemysubmarine":
				return new String[]{"leftofsubmarine", "rightofsubmarine", "aboveofsubmarine", "belowofsubmarine", "nearbysubmarine"};
			case "missile":
				return new String[]{"leftofmissile", "rightofmissile", "aboveofmissile", "belowofmissile", "nearbymissile"};
			case "oxygen":
				return new String[]{"leftofoxygen", "rightofoxygen", "aboveofoxygen", "belowofoxygen", "nearbyoxygen"};
			default:
				return new String[]{};
		}
	}
	
	/**
	 * Get weights for an anonymous variable using the cached mappings.
	 * This is much more efficient than scanning the entire fact_weights file.
	 * 
	 * @param anonVar Anonymous variable name (e.g., "anon456")
	 * @return List of weights for this anonymous variable's object, or empty list if not found
	 */
	public java.util.List<Double> getWeightsForAnonVar(String anonVar) {
		StateObjectPair pair = anonVarCache.get(anonVar);
		if (pair == null) {
			return new java.util.ArrayList<Double>();
		}
		
		java.util.List<Double> weights = stateObjectWeights.get(pair);
		return (weights != null) ? weights : new java.util.ArrayList<Double>();
	}
	
	/**
	 * Clear all loaded weights
	 */
	public void clear() {
		factWeights.clear();
		anonVarCache.clear();
		stateObjectWeights.clear();
		weightsLoaded = false;
	}
}
