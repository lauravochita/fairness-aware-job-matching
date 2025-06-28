# Fairness-Aware Job Matching Framework

A comprehensive framework detecting bias and enhancing fairness in job-matching systems. It also provides a privacy-conserving alternative for gender inference from CV embeddings. This framework is developed under the considerations of the the EU AI Act.

## Overview

This framework provides a complete pipeline for analyzing and addressing gender bias in job-matching systems. The implementation demonstrates two components: gender inference modeling, and bias analysis with counterfactual manipulation. This work was developed as part of a Master's thesis investigating fairness in AI-powered recruitment systems.

## Framework Architecture
![Framework Architecture](assets/framework_architecture.png)

The framework consists of three main components:

The framework consists of two main components:

### Component 1: Gender Inference Models
- Baseline Variational Autoencoder (VAE) for gender classification
- Semi-supervised VAE with teacher-student learning architecture
- Cross-validation and performance evaluation
- Latent space analysis and interpretation

### Component 2: Bias Detection and Counterfactual Manipulation
- Comprehensive analysis of gender representation in search results
- Statistical significance testing with multiple correction methods
- Score distribution analysis and percentile-based evaluation
- Validation suite for result integrity
- Gender direction extraction from embedding spaces
- Counterfactual manipulation through embedding modification
- Post-intervention bias measurement and validation
- Dosage-response analysis for intervention effectiveness

## Installation and Setup

### Requirements

```
torch>=1.9.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
tqdm>=4.60.0
faiss-cpu>=1.7.0
```

### Installation

```bash
git clone https://github.com/lauravochita/fairness-aware-job-matching.git
cd fairness-aware-job-matching
pip install -r requirements.txt
pip install -e .
```

## Usage Examples

### Basic Configuration

```python
from fairness_framework import Config
from fairness_framework.utils.config import ModelConfig

# Configure data paths
Config.update_base_path("/path/to/data/directory")

# Access configuration
embedding_files = Config.get_embedding_files()
model_params = ModelConfig()
```

### Component 1: Semi-Supervised VAE for Gender Inference

```python
from fairness_framework.models import run_semi_supervised_gender_vae_fixed
from fairness_framework.models.semi_supervised_vae import ImprovedSemiSupervisedVAE

# Run complete semi-supervised VAE pipeline
trained_model, baseline_model, results_df, improvement = run_semi_supervised_gender_vae_fixed()

# Create custom model
model = ImprovedSemiSupervisedVAE(
    feat_d=embedding_dimension,
    feat_related_d=embedding_dimension,
    label_dim=1,
    hidden_dim=512,
    z_dim=32,
    a_dim=2
)
```

### Component 2 a.: Bias Detection

```python
from fairness_framework.bias_analysis import run_bias_analysis
from fairness_framework.bias_analysis.statistical_tests import comprehensive_statistical_validation

# Run comprehensive bias analysis
bias_results = run_bias_analysis(
    test_job_queries=test_queries,
    gender_mapping=gender_labels,
    resume_corpus_ids=corpus_ids,
    create_faiss_index=index_function,
    search_faiss=search_function
)

# Perform statistical validation
validation_results = comprehensive_statistical_validation(
    search_results=bias_results['search_results'],
    gender_mapping=gender_labels,
    population_proportions={'male': 0.558, 'female': 0.442}
)
```

### Component 2 b.: Counterfactual Analysis

```python
from fairness_framework.counterfactual import comprehensive_gender_classifier_analysis
from fairness_framework.counterfactual.manipulation import flip_embeddings_along_gender_direction
from fairness_framework.counterfactual.post_intervention_bias import run_counterfactual_bias_analysis

# Analyze gender classifier
classifier_results = comprehensive_gender_classifier_analysis(
    embeddings=embedding_data,
    labels=gender_labels,
    visualize=True
)

# Apply counterfactual intervention
modified_embeddings = flip_embeddings_along_gender_direction(
    embeddings=original_embeddings,
    gender_weights=classifier_results['gender_weights'],
    flip_factor=0.5
)

# Evaluate intervention effectiveness
counterfactual_results = run_counterfactual_bias_analysis(
    gender_mapping=gender_labels,
    run_bias_analysis_jobs_to_resumes_func=bias_analysis_function
)
```

## Data Requirements

The framework expects the following data structure:

```
data/
├── complete_job_matching_embeddings/
│   ├── jobs.npy
│   ├── resumes.npy
│   ├── jobs_as_queries.npy
│   ├── cvs_as_passages.npy
│   └── embedding_mapping.csv
├── Datasets/
│   └── eval_dataset.csv
└── sample_jobs_100.json
```

### Input Data Formats

- **Unformatted Embeddings**: NumPy arrays (.npy) containing dense vector representations of candidate CVs (resumes.npy)
- **Formatted embeddings**: NumPy arrays (.npy) containing dense vector representations of the same candidate CVs but task-formatted as passages for the search (cvs_as_passages.npy)
- **Gender Labels**: CSV file with candidate IDs and oracle gender labels (eval_dataset.csv)
- **Query Data**: JSON files containing test job descriptions and IDs (sample_jobs_100.json)
#### Note: The used datasets are not available in this GitHub repository as they are proprietary to the interniship company and contain sensitive information about the candidates.
  

## Demonstration Notebooks

Two comprehensive Jupyter notebooks demonstrate the framework capabilities:

- `notebooks_demo/Bias_Analysis_Counterfactual_Manipulation.ipynb`: Walkthrough of bias detection with statistical analysis, and counterfactual manipulation with post-intervention analysis
- `notebooks_demo/Semi_Supervised_VAE_Gender_Inference.ipynb`: Semi-Supervised VAE model training and evaluation pipeline
#### Note: These notebooks serve as demos for how to implement the modular code base. They represent a (partial) re-run of the original implementation.
  

## Important Limitations

### Proprietary System Dependencies

This framework integrates with proprietary job-matching system components that are not publicly available:

- **BiEncoder Architecture**: Proprietary dual-encoder model for job-resume matching
- **FAISS Integration**: Custom search implementation with proprietary optimizations
- **Model Utilities**: Specialized serialization and inference components

**Academic Use**: The framework architecture, methodology, and analysis techniques can be adapted to other embedding-based matching systems for research purposes only.

**Reproduction**: While the exact implementation requires proprietary components, the statistical methods, model architectures, and bias detection techniques can be applied to publicly available embeddings and datasets for research purposes only.

### Data Sensitivity

The framework was developed using real recruitment data with appropriate privacy protections. 

## Technical Implementation

### Key Features

- **Modular Design**: Independent components that can be used separately or together
- **Statistical Rigor**: Multiple testing correction, confidence intervals, and effect size calculations
- **Robust Validation**: Cross-validation, permutation tests, and bootstrap sampling
- **Comprehensive Analysis**: Multi-faceted bias detection across different ranking depths
- **Intervention Validation**: Systematic evaluation of post-intervention results

## Code Structure

```
fairness_framework/
├── __init__.py
├── utils/
│   ├── config.py
│   └── proprietary_imports.py
├── data/
│   └── data_loader.py
├── bias_analysis/
│   ├── bias_detector.py
│   ├── statistical_tests.py
│   └── validation.py
├── models/
│   ├── vae_baseline.py
│   └── semi_supervised_vae.py
└── counterfactual/
    ├── gender_classifier.py
    ├── manipulation.py
    └── post_intervention_bias.py
```

## Academic Context

This framework was developed as part of a Master's thesis investigating fairness in AI-powered recruitment systems. The work contributes to the growing body of research on algorithmic bias detection and mitigation in high-stakes applications.

## Citation

If you reference this work in academic publications, please cite:

```
Towards Fairness-Aware Job-Matching Systems in the EU: A Framework for Sensitive Attribute Inference And Counterfactual Manipulation Under the AI Act
Laura Vochita, 2025
GitHub: https://github.com/lauravochita/fairness-aware-job-matching
```

## Contact

Laura Vochita  
GitHub: [@lauravochita](https://github.com/lauravochita)
E-mail: laura.vochita@student.uva.nl


## License

This project is provided for academic and research purposes. The framework architecture and methodology are available under an open license, while integration with proprietary systems requires separate licensing agreements.
