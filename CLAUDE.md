# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This is a machine learning learning repository based on Sebastian Raschka's "Machine Learning with PyTorch and Scikit-Learn". The project uses Python with conda/pip for dependency management.

### Setting up the environment:
```bash
# Method 1: Using requirements.txt
conda create -n raschka-ml python=3.9 -y
conda activate raschka-ml
pip install -r requirements.txt

# Method 2: Direct conda installation
/opt/anaconda3/bin/conda create -n raschka-ml python=3.9 numpy=1.21.2 scipy=1.7.0 scikit-learn=1.0 matplotlib=3.4.3 pandas=1.3.2 jupyter seaborn pytorch torchvision -c conda-forge -c pytorch -y
```

### Checking the environment:
```bash
python chapters/python_environment_check.py
```

## Repository Architecture

The repository follows a chapter-based learning structure:

```
├── chapters/
│   ├── python_environment_check.py     # Environment validation script
│   └── ch{XX}_{topic}/                 # Chapter directories
│       ├── notes.md                    # Learning notes and key concepts
│       ├── book_examples.ipynb         # Jupyter notebook with book examples
│       ├── {topic}.data               # Chapter-specific datasets
│       └── mini_project/              # Hands-on practice projects
├── datasets/                          # Shared practice datasets
└── projects/                          # Larger integrated projects
```

Each chapter is self-contained with:
- **notes.md**: Key concepts, formulas, and personal insights
- **book_examples.ipynb**: Jupyter notebooks implementing book examples
- **mini_project/**: Applied learning projects using chapter concepts
- Data files (e.g., iris.data) for chapter-specific exercises

## Common Development Tasks

### Working with Jupyter notebooks:
```bash
# Start Jupyter server
jupyter notebook

# Run specific notebook
jupyter notebook chapters/ch02_training_basics/book_examples.ipynb
```

### Environment validation:
```bash
# Check if all required packages are installed with correct versions
python chapters/python_environment_check.py
```

## Project Conventions

- Each chapter follows the same structure for consistency
- Jupyter notebooks are the primary format for interactive learning and experimentation
- Data files are stored alongside their respective chapters
- Personal experiments and variations are documented in mini_project directories
- All code follows the learning progression from basic concepts to advanced implementations