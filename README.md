# Machine Learning Journey

This repository contains my learning journey through machine learning concepts, implementations, and projects based on Sebastian Raschka's "Machine Learning with PyTorch and Scikit-Learn".

## Repository Structure

```
ml-learning-journey/
├── README.md
├── requirements.txt
├── chapters/
│   ├── ch02_training_basics/
│   │   ├── notes.md
│   │   ├── book_examples.py
│   │   ├── my_experiments.py
│   │   └── mini_project/
│   ├── ch03_classification/
│   │   ├── notes.md
│   │   ├── book_examples.py
│   │   ├── my_experiments.py
│   │   └── mini_project/
│   └── ...
├── datasets/
│   └── (your practice datasets)
└── projects/
    └── (larger integrated projects)
```

## Getting Started

1. **Install Dependencies**

Method 1: Using requirements.txt:

```
conda create -n raschka-ml python=3.9 -y
conda activate raschka-ml
pip install -r requirements.txt
```

Method 2: Direct conda command:

```
/opt/anaconda3/bin/conda create -n raschka-ml python=3.9 numpy=1.21.2 scipy=1.7.0
scikit-learn=1.0 matplotlib=3.4.3 pandas=1.3.2 jupyter seaborn pytorch torchvision -c
conda-forge -c pytorch -y
```

1. **Navigate to Chapters**
   Each chapter contains:
   - `notes.md`: Key concepts and personal notes
   - `book_examples.py`: Code examples from the book
   - `my_experiments.py`: Personal implementations and variations
   - `mini_project/`: Hands-on projects applying chapter concepts

## Chapters

- **Chapter 2**: Training Basics
- **Chapter 3**: Classification
- _(More chapters to be added as I progress)_

## Datasets

The `datasets/` directory contains practice datasets used throughout the learning journey.

## Projects

The `projects/` directory contains larger, integrated projects that combine concepts from multiple chapters.

## Learning Approach

- Follow along with book examples
- Implement concepts from scratch
- Experiment with variations and improvements
- Build mini-projects to solidify understanding
- Document key insights and learnings

## Progress Tracking

This repository serves as both a learning tool and a portfolio of my machine learning journey, showcasing progression from basic concepts to advanced implementations.

```

```
