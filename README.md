<img src="docs/figures/cocoon_logo_transparent.png" width="128"/> 


# cocoon
**co**upled **co**mmunication simulation with an **on**line trained meta-model

## Overview
cocoon (**co**upled **co**mmunication simulation with an **o**nline trained meta-model) is a novel approach to 
approximating communication simulations in Cyber-Physical Energy Systems (CPES). 
This approach addresses the computational challenges of simulating communication networks in CPES by introducing an 
online meta-modeling approach that progressively adapts to specific communication patterns during simulation execution.

## Key Features

- **Graph-based network representation** that captures essential topological and state information without requiring explicit network topology inputs
- **Four-phase adaptive learning methodology** (EGG, LARVA, PUPA, BUTTERFLY) for progressively improving prediction accuracy
- **Online integration capabilities** with energy system simulations
- **Dynamic adaptation** to different communication scenarios
- **Computational efficiency** without sacrificing accuracy

## Methodology

Our approach follows a four-phase process inspired by the metamorphosis of a butterfly:

### 1. EGG Phase 
- Establishes foundation for prediction through cluster analysis and pre-training
- Analyzes historical message data to identify distinct patterns
- Uses hierarchical clustering with optimized decision tree regressors
- [egg.py](src/training/egg.py)

### 2. LARVA Phase
- Marks the beginning of simulation scenario execution
- Addresses initialization bias by gathering sufficient data
- Assigns new messages to the closest historical cluster
- - [larva.py](src/training/larva.py)

### 3. PUPA Phase
- Trains an additional regressor online on current scenario data
- Combines cluster-based and online predictions through a weighting mechanism
- Progressively improves prediction accuracy as the scenario unfolds
- [pupa.py](src/training/pupa.py)

### 4. BUTTERFLY Phase
- Activated when weighted predictions consistently meet a predefined accuracy threshold
- Replaces the original communication simulation entirely
- Continues to process messages using the weighted prediction approach

## Usage
Users can run the simple use case provided in the repository:

```bash
# Run the example Storage Application scenario
python src/analysis/00_paper_usecase.py
```

This script demonstrates the complete workflow of the cocoon approach using a Storage Application scenario.


## Repository Structure
```.
├── cocoon/                         # Main package
│   ├── docs/                       # Documentation
│   ├── src/                        # Source code
│   │   ├── analysis/               # Analysis and evaluation scripts
│   │   │   └── 00_paper_usecase.py # Paper use case implementation
│   │   └── training/               # Implementation of training phases
│   │       ├── egg.py              # EGG phase implementation
│   │       ├── larva.py            # LARVA phase implementation
│   │       └── pupa.py             # PUPA phase implementation
│   ├── utils/                      # Utility functions
│   ├── cocoon.py                   # Main implementation
│   ├── communication_network_graph.py  # Graph-based network representation
│   ├── events.py                   # Event handling
│   ├── scenario_configuration.py   # Configuration for scenarios
│   └── state_definitions.py        # Definition of state attributes
├── state_data/                     # State data from simulation scenarios
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Installation

```bash
git clone https://github.com/OFFIS-DAI/cocoon.git
cd cocoon
pip install -r requirements.txt
```

## Example Results

Our experimental results with a Storage Application scenario demonstrated significant improvements in prediction accuracy:

- Cluster-based predictions: Mean absolute error of 2.78ms
- Online-trained regressor: Mean absolute error of 0.62ms
- Weighted predictions: Mean absolute error of only 0.28ms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Malin Radtke - malin.radtke@offis.de