# RDNBoost Split Scoring and Node Selection - ICML Dataset Example

## Overview

RDNBoost learns regression trees by iteratively selecting the best literal (split) to add to a clause. The process uses **beam search** with **variance-based scoring** to build optimal tree structures.

---

## 1. The Search Process

### Initial State
```
Root clause: CoAuthor(_, _)
Covers: 325 positive examples
Score: -Infinity (no split yet, just matches everything)
```

### Beam Search Algorithm
1. **Start with root node** (empty clause body)
2. **Generate candidates**: Try adding each possible literal from mode declarations
3. **Score each candidate**: Compute weighted variance for the split
4. **Keep best nodes**: Maintain a beam of promising candidates
5. **Expand recursively**: Add more literals to best candidates
6. **Terminate**: When variance is low enough or max depth reached

---

## 2. Scoring Formula

### Weighted Variance Calculation

For a split with TRUE and FALSE branches:

```
TOTAL SCORE = weighted_variance(TRUE) + weighted_variance(FALSE)
```

Where for each branch:

```java
weighted_variance = sumOfOutputSquared - (sumOfOutputAndNumGrounding² / sumOfNumGroundingSquared)
```

**Components:**
- `sumOfOutputSquared` = Σ(output² × weight) for all examples
- `sumOfOutputAndNumGrounding` = Σ(numGroundings × output × weight)
- `sumOfNumGroundingSquared` = Σ(numGroundings² × weight)

**Key Point:** Lower score = better split (less variance = more homogeneous branches)

---

## 3. Real Example from ICML Dataset

### First Iteration: Finding Best Root Split

**Starting examples:** 325 CoAuthor pairs

#### Candidate 1: Split by Berkeley Affiliation
```
Split: CoAuthor(A, _) :- Affiliation(A, "University_of_California_Berkeley")

TRUE branch:  60 examples (both authors, one is from Berkeley)
  weighted_variance = 13.48

FALSE branch: 265 examples (neither from Berkeley)  
  weighted_variance = 58.26

TOTAL SCORE: 71.74
```

#### Candidate 2: Split by MIT Affiliation
```
Split: CoAuthor(_, A) :- Affiliation(A, "Massachusetts_Institute_of_Technology")

TRUE branch:  26 examples
  weighted_variance = 6.32

FALSE branch: 299 examples
  weighted_variance = 64.45

TOTAL SCORE: 70.77  ← BETTER! (lower)
```

#### Candidate 3: Split by Research Topic
```
Split: CoAuthor(A, _) :- ResearchTopic(A, "Segmentation")

TRUE branch:  6 examples (work on image segmentation)
  weighted_variance = 0.67

FALSE branch: 319 examples
  weighted_variance = 69.08

TOTAL SCORE: 69.74  ← BEST! Selected as first clause
```

**Winner:** Research topic "Segmentation" because it has the **lowest total variance** (69.74)

---

## 4. Why This Split is Good

### Understanding the Numbers

The best split has:
- **Very low variance in TRUE branch** (0.67): All 6 segmentation researchers have similar collaboration patterns
- **Small TRUE branch** (6 examples): Highly specific, pure group
- **Total variance** lower than other candidates

### What Variance Measures

Variance in a branch = how spread out the **gradient values** are:
- **Low variance**: Examples have similar target values (gradients) → good to group together
- **High variance**: Examples have different target values → bad grouping

For CoAuthor prediction:
- TRUE branch (works on segmentation): Mostly positive examples, high collaboration rate
- FALSE branch (doesn't work on segmentation): Mixed, requires further splitting

---

## 5. Recursive Tree Building

After selecting the first split, RDNBoost continues:

### Tree After First Split
```
CoAuthor(A, B) :- ResearchTopic(A, "Segmentation")
  Covers: 6 positive examples
  Leaf value: 0.8 (high probability of co-authorship)
```

Remaining examples: 319

### Second Iteration: Split Remaining Examples

Now search for best split on the 319 remaining examples:

```
Best candidates:
- Affiliation(A, "University_of_California_Berkeley") → score: 69.41
- Affiliation(A, "EPFL") → score: 69.11
- ResearchTopic(A, "Cross_entropy") → score: 67.70  ← BEST
```

**Winner:** Cross-entropy topic (score 67.70)

### Growing the Tree Deeper

After selecting 2nd split, can add MORE literals to the first clause:

```
Original: CoAuthor(A, B) :- ResearchTopic(A, "Segmentation")
Extend to: CoAuthor(A, B) :- ResearchTopic(A, "Segmentation"), 
                              ResearchTopic(B, "Computer_vision")
```

This creates a **conjunction** (AND) of conditions, making the rule more specific.

---

## 6. Complete Scoring Example with Actual Values

Let's trace one split in detail:

### Split: `CoAuthor(A, _) :- ResearchTopic(A, "Affine_transformation")`

**TRUE Branch (48 examples):**
```
Example 1: output=0.35, weight=1.0, numGroundings=1
Example 2: output=0.42, weight=1.0, numGroundings=1
...
Example 48: output=0.38, weight=1.0, numGroundings=1

Calculations:
sumOfOutputSquared = 0.35² + 0.42² + ... + 0.38² = 8.24
sumOfOutputAndNumGrounding = (1×0.35) + (1×0.42) + ... + (1×0.38) = 18.72
sumOfNumGroundingSquared = 1² + 1² + ... + 1² = 48.0

weighted_variance = 8.24 - (18.72² / 48.0)
                  = 8.24 - (350.44 / 48.0)
                  = 8.24 - 7.30
                  = 0.94
```

**FALSE Branch (277 examples):**
```
Similar calculation → weighted_variance = 63.88
```

**Total Score:** 0.94 + 63.88 = **64.82**

This was one of the BEST splits found (very low score)!

---

## 7. Key Insights

### Why Variance Works for Regression Trees

1. **Homogeneity**: Low variance means examples in a branch are similar
2. **Prediction quality**: Similar examples → accurate mean/median prediction
3. **Gradient boosting**: In boosting, we're fitting residuals (gradients), so grouping similar residuals is crucial

### How Beam Search Helps

- **Explores many candidates**: Tries 100+ possible literals per iteration
- **Keeps best**: Only expands most promising nodes
- **Avoids local minima**: Maintains multiple paths in the beam
- **Efficient**: Doesn't explore every possible tree

### The Role of Modes

Mode declarations tell RDNBoost what literals to try:
```
mode: Affiliation(+name, #university)
  → Try: Affiliation(A, "MIT"), Affiliation(A, "Berkeley"), etc.

mode: ResearchTopic(+name, #topic)  
  → Try: ResearchTopic(A, "AI"), ResearchTopic(A, "Vision"), etc.
```

The `+` and `#` symbols indicate:
- `+`: Variable already bound (input)
- `#`: Constant to be specified (output, enumerate all values)
- `-`: Variable to be generated (output, create new variable)

---

## 8. Summary: Complete Algorithm

```
1. Start with root: CoAuthor(_, _) covering all examples

2. FOR each tree depth:
   a. Generate candidate literals from modes
   b. FOR each candidate:
      - Split examples into TRUE/FALSE branches
      - Calculate weighted variance for each branch
      - Score = variance_true + variance_false
   c. Select best split (lowest score)
   d. Add as a clause to the tree
   e. Update example coverage (remove covered examples)

3. STOP when:
   - All examples covered
   - Max tree size reached
   - Variance below threshold
```

**Final Result:** A set of clauses (tree paths) that predict CoAuthor relationships based on affiliations and research topics.
