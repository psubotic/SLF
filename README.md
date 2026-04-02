# SLF Detector 
[![SLF Detector Tests](https://github.com/psubotic/SLF/actions/workflows/slf-tests.yaml/badge.svg)](https://github.com/psubotic/SLF/actions/workflows/slf-tests.yaml)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)](https://github.com/psubotic/SLF)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



A PoC for detecting and counting adult Lycorma delicatula a.k.a Spotted Lanternfly(SLF) in pheromone sticky trap photographs. The challenge of this project was to find a model/pipeline that has reasonable accuracy with: 
1. No labeled training dataset,
2. Deployment on a CPU (no GPU).

## Quick Start
```bash
git clone https://github.com/psubotic/SLF.git
cd SLF
pip install -r requirements.txt

# Run demo with synthetic data
python scripts/run_demo.py --mode synthetic
```

## Our Approach
Four-stage classical computer vision pipeline:
1. **Preprocessing** - CLAHE contrast enhancement, glare removal
2. **Region Proposal** - MSER + Canny contours with NMS
3. **Heuristic Filter** - Color, aspect ratio, and solidity rules
4. **Descriptor Classifier** - HOG+LBP features + Random Forest


## Performance
| Metric | Heuristic-only | **Our Approach** |
|--------|----------------|-----------------|
| Precision | 41% | **80%** |
| Recall | 91% | **83%** |
| F1 | 56% | **81%** |


