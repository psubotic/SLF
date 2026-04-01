# SLF Trap Detector 
[![SLF Detector Tests](https://github.com/psubotic/SLF/actions/workflows/slf-tests.yaml/badge.svg)](https://github.com/psubotic/SLF/actions/workflows/slf-tests.yaml)

[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)](https://github.com/psubotic/SLF)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



A PoC for detecting and counting adult Lycorma delicatula a.k.a Spotted Lanternfly(SLF) in pheromone sticky trap photographs. The challenge of this project was to find a model/pipeline that has reasonable accuracy with: 
1. no labeled training dataset,
2. deployment on a CPU (no GPU).

