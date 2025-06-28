# Proprietary Dependencies Documentation

## Overview

This framework integrates with several proprietary components that are **NOT** included in this public repository. These components were developed at a private company and contain sensitive intellectual property.

## Missing Proprietary Components

### 1. **BiEncoder Job-Matching System**
- **BiEncoderConfig**: Configuration class for dual-encoder models
- **BiEncoder**: Proprietary neural architecture for job-resume matching
- **Model utilities**: Specialized serialization functions (`numpy_serialize`, etc.)

**Required files (not included):**
```
proprietary_system/
├── inference_endpoint/src/
│   ├── types.py          # BiEncoderConfig
│   ├── model.py          # BiEncoder class
│   └── utils.py          # Helper functions
└── fine_tuning/src/
    └── tools/
        └── evaluation_utils.py  # FAISS utilities
```

### 2. **FAISS Search Implementation**
- **create_faiss_index()**: Custom FAISS index creation with proprietary optimizations
- **search_faiss()**: Optimized search function for job-resume matching

### 3. **FairWS VAE Components**
- **VAE_fair**: Original VAE implementation from FairWS framework
- **latent_loss**: Custom loss functions for fairness-aware training

**Available from:** [FairWS GitHub Repository](https://github.com/huaishengzhu/FairWS)

**Required file:**
```
FairWS-main/
└── VAE.py              # VAE_fair class and latent_loss
```

**Installation:**
```bash
git clone https://github.com/huaishengzhu/FairWS.git
# Place in your data directory or update Config.BASE_DATA_PATH
```

## Impact on Reproducibility

### What Works Without Proprietary Components:
**Statistical Analysis Methods**: All bias detection algorithms, statistical tests, and validation techniques  
**VAE Architectures**: Model architectures and training procedures (with adaptations)  
**Counterfactual Methodology**: Mathematical framework for embedding manipulation  
**Framework Structure**: Complete modular design and API  
**FairWS Integration**: Available from [public repository](https://github.com/huaishengzhu/FairWS)

### What Requires Adaptations:
**Embedding Generation**: Need alternative embedding models (e.g., SentenceTransformers, OpenAI embeddings)  
**Search Implementation**: Need alternative FAISS setup or other vector search  

### What Cannot Be Reproduced:
**Exact Results**: Proprietary embeddings and search optimizations  
**Full Pipeline**: End-to-end system requires custom implementations  
**Original Data**: Real recruitment data is confidential  

## Alternative Implementation Guide

### For Academic Research:

1. **Replace BiEncoder with Open Alternatives:**
```python
# Instead of proprietary BiEncoder
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
job_embeddings = model.encode(job_descriptions)
cv_embeddings = model.encode(cv_texts)
```

2. **Use Standard FAISS:**
```python
import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product
    index.add(embeddings.astype('float32'))
    return index

def search_faiss(index, queries, k=50):
    scores, indices = index.search(queries.astype('float32'), k)
    return scores, indices
```

3. **Setup FairWS Components:**
```bash
# Clone FairWS repository
git clone https://github.com/huaishengzhu/FairWS.git

# Update your config to point to FairWS location
from fairness_framework.utils.config import Config
Config.update_base_path("/path/to/your/data")  # Should contain FairWS-main/
```

### Licensing and Legal Considerations:

- **This framework**: Open for academic research under provided license
- **Proprietary components**: Require separate licensing agreements
- **Company data**: Not available due to privacy and confidentiality
- **Commercial use**: Prohibited without explicit permission

## Contact for Access

For researchers who need access to proprietary components for legitimate academic research:

1. Contact the original company (details available upon request)
2. Sign appropriate research agreements
3. Obtain institutional review board approval if working with sensitive data
