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
