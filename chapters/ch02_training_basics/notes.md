# Chapter 2: Training Basics

## Key Concepts

## Notes

I had to adjust the `np.where (y == 'Iris-setosa', -1, 1)` to use -1 and 1. When I used 0,1 in the `where()` call, the perceptron was not converging.

With using labels from book's example: (0, +1):

- When actual=0, predicted=+1: update = 0.1 × (0 - 1) = -0.1 ✓
- When actual=+1, predicted=+1: update = 0.1 × (1 - 1) = 0 ✓
- When actual=0, predicted=-1: update = 0.1 × (0 - (-1)) = +0.1 ✓
- When actual=+1, predicted=-1: update = 0.1 × (1 - (-1)) = +0.2

The missing symmetry for the last case was causing an issue. Changing the where() to use -1 and 1 in the code fixed the convergence.

```python
y = np.where(y == 'Iris-setosa', -1, 1)
```

## Plotting Explanation

### The Big Picture

We want to create a plot that shows:

1. **Background colors** = what the perceptron would predict for ANY point
2. **Decision boundary** = the line where predictions change from one class to another
3. **Actual data points** = your training data on top

---

#### Step 1: Create a Grid of Test Points

Think of this like creating a fine mesh or net over your entire plot area:

```python
# Let's say your data ranges from:
# Sepal length: 4.0 to 7.0
# Petal length: 1.0 to 5.0

x1_min, x1_max = 4.0, 7.0  # Sepal length range (with padding)
x2_min, x2_max = 1.0, 5.0  # Petal length range (with padding)

# Create arrays of evenly spaced points
x1_points = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, ...]  # Every 0.2 units
x2_points = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, ...]  # Every 0.2 units

# np.meshgrid creates ALL COMBINATIONS of these points
xx1, xx2 = np.meshgrid(x1_points, x2_points)

What meshgrid does: Creates two grids where every (i,j) position represents one test
point:

# If x1_points = [4.0, 4.2, 4.4] and x2_points = [1.0, 1.2, 1.4]

xx1 = [[4.0, 4.2, 4.4],    # Row 0: sepal length values
       [4.0, 4.2, 4.4],    # Row 1: same sepal length values
       [4.0, 4.2, 4.4]]    # Row 2: same sepal length values

xx2 = [[1.0, 1.0, 1.0],    # Row 0: petal length = 1.0 for all
       [1.2, 1.2, 1.2],    # Row 1: petal length = 1.2 for all
       [1.4, 1.4, 1.4]]    # Row 2: petal length = 1.4 for all

The test points are:
- Position (0,0): sepal=4.0, petal=1.0
- Position (0,1): sepal=4.2, petal=1.0
- Position (0,2): sepal=4.4, petal=1.0
- Position (1,0): sepal=4.0, petal=1.2
- ...and so on

---
Step 2: Convert Grid to List of Points

# Flatten the grids into 1D arrays
xx1_flat = xx1.ravel()  # [4.0, 4.2, 4.4, 4.0, 4.2, 4.4, 4.0, 4.2, 4.4]
xx2_flat = xx2.ravel()  # [1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.4, 1.4, 1.4]

# Stack them vertically then transpose
points_array = np.array([xx1_flat, xx2_flat]).T

# Result: each row is one test point [sepal_length, petal_length]
points_array = [[4.0, 1.0],  # Test point 1
                [4.2, 1.0],  # Test point 2
                [4.4, 1.0],  # Test point 3
                [4.0, 1.2],  # Test point 4
                [4.2, 1.2],  # Test point 5
                [4.4, 1.2],  # Test point 6
                [4.0, 1.4],  # Test point 7
                [4.2, 1.4],  # Test point 8
                [4.4, 1.4]]  # Test point 9

---
Step 3: Get Predictions for All Test Points

# Ask the perceptron: "What class would you predict for each test point?"
predictions = classifier.predict(points_array)

# Example result:
predictions = [-1, -1, 1, -1, 1, 1, 1, 1, 1]
#               ↑   ↑  ↑   ↑   ↑  ↑  ↑  ↑  ↑
#              pt1 pt2 pt3 pt4 pt5 pt6 pt7 pt8 pt9

What this means:
- Test point 1 (4.0, 1.0) → Predicted class: -1 (Setosa)
- Test point 2 (4.2, 1.0) → Predicted class: -1 (Setosa)
- Test point 3 (4.4, 1.0) → Predicted class: +1 (Versicolor)
- ...and so on

---
Step 4: Reshape Back to Grid for Plotting

# Convert the flat list back to the original grid shape
predictions_grid = predictions.reshape(xx1.shape)

# Result:
predictions_grid = [[-1, -1,  1],    # Row 0: petal length 1.0
                    [-1,  1,  1],    # Row 1: petal length 1.2
                    [ 1,  1,  1]]    # Row 2: petal length 1.4

---
Step 5: Plot the Results

# Create a contour plot that colors regions based on predictions
plt.contourf(xx1, xx2, predictions_grid, alpha=0.3, cmap=cmap)

What contourf does:
- Takes the grid coordinates (xx1, xx2) and the predictions
- Colors each grid cell based on the prediction value
- -1 predictions → Red regions
- +1 predictions → Blue regions
- The boundary between colors = decision boundary!

# Finally, plot your actual training data on top
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0],      # Sepal length for this class
                y=X[y == cl, 1],      # Petal length for this class
                alpha=0.8,
                c=colors[idx],         # Color for this class
                marker=markers[idx],   # Shape for this class
                label=f'Class{cl}')    # Legend label

---
The Final Result

You get a plot where:
- Red background = "If a flower had these measurements, I'd predict Setosa"
- Blue background = "If a flower had these measurements, I'd predict Versicolor"
- Boundary line = Where the perceptron switches its prediction
- Actual dots = Your real training data points

This shows you visually how well your perceptron learned to separate the two types of flowers!
```

## Gradient Descent

Gradient descent finds the best weights by minimizing the loss function by taking a step in the opposite direction of the loss gradient that is calculated from the whole training dataset.

- Also referred to as full batch gradient descent
- Costly when batch size is large

  In Matrix Form (Adaline Implementation)

  w := w + η _ (2/n) _ X^T _ (y - ŷ)
  b := b + η _ (2/n) \* Σ(y - ŷ)

## Stochastic Gradient Descent (SGC)

AdalineSGD implements Stochastic Gradient Descent version of Adaline

- updates weights one sample at a time instead of using the entire
  batch.

Alternative to full batch gradient descent (iterative online gradent descent)

- Important to shuffle data so training data is in random order
- Use case: Continue training with new data without starting over!

  Key Differences: SGD vs Batch

  | Aspect                | Batch Adaline | SGD Adaline            |
  | --------------------- | ------------- | ---------------------- |
  | Updates per epoch     | 1             | n (number of samples)  |
  | Memory usage          | High          | Low                    |
  | Convergence           | Smooth        | Noisy but often faster |
  | Online learning       | No            | Yes (partial_fit)      |
  | Data order dependency | No            | Yes (needs shuffling)  |

  Training Flow

w := w - η \* ∂L(xi, yi)/∂w

Where:

- w = weights
- η = learning rate
- ∂L(xi, yi)/∂w = gradient of loss for single sample (xi, yi)
- := means "update to"

## Important Formulas

## Questions/Clarifications
