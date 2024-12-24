# Reinforcement Learning Course 
Codebase for the [Reinforcement Learning Course](https://www.edreate.com/courses/reinforcement-learning/) . This repository provides the hands-on codebase to accompany the lessons.For an introduction to the **Lunar Lander v3**  environment used in this course, refer to this [lesson](https://edreate.com/courses/reinforcement-learning/gynasium-environments-for-reinforcement-lerning/lunar-lander/) .

---


## Repository Overview 
This repository is structured to teach reinforcement learning (RL) concepts through hands-on coding, focusing on building and testing RL algorithms using **Lunar Lander v3**  from the Gymnasium library.
Key objectives include:
 
1. Training a **PolicyNetwork**  using various RL algorithms.
 
2. Comparing the strengths and weaknesses of **on-policy**  and **off-policy**  approaches.

3. Using a unified neural network architecture to ensure fair comparisons across algorithms.

4. Visualizing and testing trained policies with the provided tools.


---


## Algorithms Covered 

The course explores a range of RL algorithms, including:
**On-Policy Algorithms** 
- Vanilla Policy Gradient (VPG)
- Proximal Policy Optimization (PPO)

**Off-Policy Algorithms** 
- Deep Q-Learning (DQN)
- Soft Actor-Critic (SAC)

All algorithms are implemented from scratch using PyTorch and grounded in RL theory.


---


## Repository Components 

1. **PolicyNetwork** A shared neural network architecture (`PolicyNetwork`, defined in `models.py`) is used to ensure consistency across algorithms. 

- **Input** : Current state of the environment.
 
- **Output** :
  - Action probabilities (on-policy algorithms).

  - Q-values (off-policy algorithms).

2. **Training Scripts** Jupyter notebooks are provided for training the `PolicyNetwork` with each algorithm. Notebooks include:

- Step-by-step implementation of each algorithm.

- Explanations of key concepts.

- Code for saving trained policies.
For in-depth theoretical explanations, visit the [Edreate course page](https://edreate.com/courses/reinforcement-learning/) .

3. **Testing and Visualization** The `lunar_lander_in_action.py` script is included to:

- Load trained policy.
 
- Visualize performance in the **Lunar Lander v3**  environment.

- Display metrics like rewards, timesteps, and lander dynamics.


---


## How to Use 

### Prerequisites 
Ensure you have **Python 3.9**  or higher. Install dependencies with:

```bash
pip install torch swig gymnasium[box2d] pygame numpy
```

### Training the PolicyNetwork 

Follow the Jupyter notebooks provided in the repository. Each notebook corresponds to a specific algorithm.

### Visualizing Trained Policies 

Run the visualization script:


```bash
python lunar_lander_in_action.py
```

### Extending the Codebase 

You can:

- Adjust training hyperparameters.

- Add new RL algorithms.

- Modify the visualization script for enhanced insights.


---


## Learning Goals 

By engaging with this repository, you will:
 
1. Understand the distinctions between **on-policy**  and **off-policy**  algorithms.

2. Implement RL algorithms step-by-step using PyTorch.

3. Gain experience solving RL tasks with Gymnasium environments.

4. Learn how to debug, visualize, and optimize RL policies.


---


## Acknowledgments 

This repository was created to help RL enthusiasts gain practical knowledge and implement foundational concepts. It is inspired by the open-source community and designed to accelerate your learning journey.

Happy learning! 🚀


# Reinforcement Learning Course

This repository contains materials for the Reinforcement Learning course, including code for training and visualizing models in environments like Lunar Lander.

## Setup Instructions

Follow these steps to set up the environment and run inference:

### 1. Clone the Repository
```bash
git clone https://github.com/edreate/Reinforcement_Learning_Course.git
cd Reinforcement_Learning_Course
```

### 2. Install Python Virtual Environment Tools (if not already installed) 


```bash
sudo apt install python3.10-venv
```

### 3. Create and Activate a Virtual Environment 


```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Required Dependencies 


```bash
pip install -r requirements.txt
```

### 5. Additional Setup 
Install the `ipython` kernel for working with Jupyter notebooks:

```bash
ipython kernel install --name "local-venv" --user
```

Install additional tools for formatting and visualization:


```bash
pip install pygame tqdm ruff
```
6. Set the `PYTHONPATH`Ensure the repository is correctly added to the `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)
```

## Running Inference 

To visualize the Lunar Lander environment in action, use the provided script:


```bash
python visualization/LunarLander_in_Action.py
```

## Code Formatting 

Run the formatting tools to maintain consistent code style:


```bash
make format
```

## Notes 
 
- Ensure you activate the virtual environment (`source venv/bin/activate`) each time you start working in the repository.
 
- The `PYTHONPATH` should be set whenever a new terminal session is started, or you can add it to your shell configuration file for persistence.

## Troubleshooting 

If you encounter issues, ensure:

- The virtual environment is activated.

- All dependencies are installed.
 
- The `PYTHONPATH` is correctly set.

Happy learning and experimenting with reinforcement learning!
