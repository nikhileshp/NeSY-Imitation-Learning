# Weighted Variance Calculation in RDNBoost - Step by Step

## Real Example from ICML Dataset

This shows **exactly** how weighted variance is calculated for a split in RDNBoost, using actual debug output from training.

---

## Split Being Evaluated

```
Split: CoAuthor(A, _) :- Affiliation(A, "Stanford_University")

TRUE branch: Authors affiliated with Stanford
FALSE branch: Authors NOT affiliated with Stanford
```

---

## TRUE Branch: 20 Examples

### Step 1: Adding Examples to Branch

Each example contributes three values to running sums:

```
Example #1: num=1, output=0.858, weight=1.0
  Contributes:
    - output²×weight = 0.858² × 1.0 = 0.736
    - num×output×weight = 1 × 0.858 × 1.0 = 0.858
    - num²×weight = 1² × 1.0 = 1.0

Example #2: num=1, output=0.858, weight=1.0
  Contributes: 0.736, 0.858, 1.0

...continuing for all 20 examples...
```

**Note:** `output` is the gradient value (difference between true label and current prediction)

### Step 2: Sum Up All Contributions

After processing all 20 examples:

```
sumOfOutputSquared = 6.133
sumOfOutputAndNumGrounding = 5.163
sumOfNumGroundingSquared = 20.0
numExamples = 20.0
```

### Step 3: Calculate Weighted Variance

Formula:
```
weighted_variance = sumOfOutputSquared - (sumOfOutputAndNumGrounding² / sumOfNumGroundingSquared)
```

Calculation:
```
term1 = sumOfOutputSquared = 6.133

term2 = (sumOfOutputAndNumGrounding)² / sumOfNumGroundingSquared
      = (5.163)² / 20.0
      = 26.656 / 20.0
      = 1.333

weighted_variance = 6.133 - 1.333 = 4.800
```

**Result:** TRUE branch weighted variance = **4.800**

---

## FALSE Branch: 309 Examples

Using the same process for the 309 Stanford authors:

```
sumOfOutputSquared = [larger value, ~210]
sumOfOutputAndNumGrounding = [sum of ~309 gradients]
sumOfNumGroundingSquared = 309.0

weighted_variance = 67.987
```

**Result:** FALSE branch weighted variance = **67.987**

---

## Total Split Score

```
TOTAL SCORE = variance(TRUE) + variance(FALSE)
            = 4.800 + 67.987
            = 72.787
```

This split gets a score of **72.787**. Lower is better!

---

## Another Example: Smaller Branch

### Split: EPFL Affiliation (13 examples)

```
TRUE Branch (13 examples):
  sumOfOutputSquared = 1.694
  sumOfOutputAndNumGrounding = 0.156
  sumOfNumGroundingSquared = 13.0
  
  term2 = (0.156)² / 13.0
        = 0.0243 / 13.0
        = 0.00187
  
  weighted_variance = 1.694 - 0.00187 = 1.692
```

**Much lower variance!** Only 1.692 vs 4.800 for Stanford.

Why? The 13 EPFL examples have **more similar gradient values**.

---

## Understanding the Formula Components

### What Each Sum Represents

1. **sumOfOutputSquared**: Total of all squared gradients
   - Measures the "energy" or magnitude of outputs
   - Formula: Σ(output² × weight)

2. **sumOfOutputAndNumGrounding**: Weighted sum of gradients
   - Like computing a weighted mean numerator
   - Formula: Σ(numGroundings × output × weight)

3. **sumOfNumGroundingSquared**: Sum of squared groundings
   - Usually just equals #examples when numGroundings=1
   - Formula: Σ(numGroundings² × weight)

### The Variance Formula Explained

```
variance = Σ(x²) - (Σx)²/n
```

This is the **sum of squares** formula for variance:
- `Σ(x²)` = sum of squared values
- `(Σx)²/n` = square of the mean
- The difference gives you variance

**In our case:**
- x = output values (gradients)
- Σ(x²) = sumOfOutputSquared
- (Σx)² = (sumOfOutputAndNumGrounding)²
- n = sumOfNumGroundingSquared

---

## Why This Works for Relational Data

### Role of numGroundings

In relational learning, a single example can "ground" (instantiate) multiple times:

```
Example: CoAuthor(A, B) where A works at MIT

If there are 5 people at MIT, this example grounds 5 times:
  - CoAuthor(Person1, B)
  - CoAuthor(Person2, B)
  - CoAuthor(Person3, B)
  - CoAuthor(Person4, B)
  - CoAuthor(Person5, B)
```

The weighted variance accounts for this:
- `numGroundings` = 5 for this example
- It contributes 5× more to the statistics
- This is why we multiply by `numGroundings` and square it

**In ICML dataset:** Most examples have `numGroundings=1` (simple binary relations)

---

## Comparing Different Splits

From actual training run:

| Split | TRUE Branch | TRUE Variance | FALSE Variance | Total Score |
|-------|-------------|---------------|----------------|-------------|
| Stanford | 20 examples | 4.800 | 67.987 | 72.787 |
| Max Planck | 27 examples | 6.667 | 65.844 | **72.511** ✓ better |
| EPFL | 13 examples | 1.692 | 70.769 | 72.461 ✓ even better |

**Winner:** EPFL has the lowest total score!

### Why EPFL Wins

- **Very homogeneous TRUE branch** (variance 1.692)
- Small but pure group of collaborators
- Even though FALSE branch has higher variance, total is still lowest

---

## The Gradient Values (Output)

### What is "output"?

```
output = gradient = (true_label - current_prediction)
```

**For first tree:**
- initial prediction = 0.5 (sigmoid(0))
- For positive example: gradient = 1.0 - 0.5 = 0.5
- For negative example: gradient = 0.0 - 0.5 = -0.5

**From the debug output:**
```
output=0.858  → positive example, current pred is ~0.142
output=-0.142 → negative example, current pred is ~0.642
```

These values change as boosting progresses and the model learns.

---

## Complete Calculation Example

Let's compute by hand for 6 examples with varied outputs:

```
Example outputs: [0.8, 0.7, 0.9, 0.6, 0.85, 0.75]
All have: num=1, weight=1.0

Step 1: Calculate sums
  sumOfOutputSquared = 0.64 + 0.49 + 0.81 + 0.36 + 0.7225 + 0.5625
                     = 3.585
  
  sumOfOutputAndNumGrounding = 0.8 + 0.7 + 0.9 + 0.6 + 0.85 + 0.75
                              = 4.6
  
  sumOfNumGroundingSquared = 1 + 1 + 1 + 1 + 1 + 1
                           = 6.0

Step 2: Calculate variance
  term2 = (4.6)² / 6.0 = 21.16 / 6.0 = 3.527
  
  variance = 3.585 - 3.527 = 0.058

Result: Very low variance! These examples are similar (all around 0.75)
```

Compare to examples with more spread: [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
- Same mean (0.5) but variance would be much higher!

---

## Key Takeaways

1. **Weighted variance measures homogeneity** - lower variance = examples more similar

2. **Formula accounts for multiple groundings** - important for relational data

3. **Split scoring is additive** - sum variance from both branches

4. **Gradients are the outputs** - not raw labels, but residuals to fit

5. **Best split minimizes total variance** - creates most homogeneous groups

This is why RDNBoost can effectively learn from relational data - it groups examples with similar gradient values together, leading to accurate predictions in each leaf.
