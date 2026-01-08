<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=47A248&center=true&vCenter=true&width=435&lines=Stepping+Stone+Solution;Transportation+Problem+Optimizer;VAM+%7C+Stepping+Stone+Algorithms;GUI+Built+with+Tkinter" alt="Typing SVG" />

  <h1>Stepping Stone Solution - Logistics Optimizer</h1>

  <p>
    <strong>An algorithmic optimization tool designed to solve complex transportation problems using Vogel's Approximation Method (VAM) and the Stepping Stone algorithm.</strong>
  </p>

  <p>
    <a href="https://stepping-stone-solution-lzezabbhmtqpyowvnesj6f.streamlit.app/"><strong>Explore the Live Demo »</strong></a>
  </p>

  [![Live Demo](https://img.shields.io/badge/Demo-Live_on_Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://stepping-stone-solution-lzezabbhmtqpyowvnesj6f.streamlit.app/)
  [![Python](https://img.shields.io/badge/Language-Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Tkinter](https://img.shields.io/badge/GUI-Tkinter-blue?style=for-the-badge)](https://docs.python.org/3/library/tkinter.html)
</div>

---

## Executive Summary
[cite_start]The Stepping Stone Solution is a desktop application developed to optimize supply chain allocation[cite: 27]. [cite_start]It automates the process of finding the most cost-effective way to transport goods from multiple sources to multiple destinations[cite: 27]. By implementing classic Operations Research algorithms, this tool reduces manual computation errors and provides an intuitive GUI for logistical decision-making.

## Core Features
* [cite_start]**Dual-Phase Optimization:** Uses Vogel's Approximation Method (VAM) for the Initial Basic Feasible Solution (IBFS) and the Stepping Stone algorithm for final optimality[cite: 27].
* [cite_start]**Dynamic Grid Support:** Optimized supply allocation for configurations up to 4 sources and 4 destinations[cite: 27].
* **Real-time Cost Calculation:** Instantly computes total transportation costs based on user-defined unit costs, supply, and demand.
* [cite_start]**User-Friendly Interface:** Built with Tkinter and Pillow for a clean, accessible desktop experience[cite: 26, 27].

---

## Technical Architecture
The application follows a structured Python architecture to separate mathematical logic from the presentation layer.



* [cite_start]**Presentation Layer:** Built using **Tkinter** for the windowing system and **Pillow** for image rendering and assets[cite: 26].
* **Algorithmic Engine:** Custom Python implementations of:
    1. **Vogel’s Approximation Method (VAM):** To find a high-quality starting solution by analyzing "penalties" for each row and column.
    2. **Stepping Stone Algorithm:** To iteratively test unoccupied cells for potential cost reductions until an optimal solution is reached.

---

## Local Development Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/AnuragGaonkar/STEPPING-STONE-SOLUTION.git](https://github.com/AnuragGaonkar/STEPPING-STONE-SOLUTION.git)
cd STEPPING-STONE-SOLUTION
```

### 2. Environment Setup

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Application:**
```bash
python Trnsprt.py
```

## Engineering Challenges and Resolutions

### Standardizing Multi-Modal Inputs
* **Challenge**: Manually calculating the closed loops required for the Stepping Stone algorithm is mathematically intensive and prone to errors in a software environment.
* **Resolution**: Developed a recursive path-finding function that identifies valid closed loops in the allocation matrix, ensuring the algorithm correctly evaluates the improvement index for every non-basic variable.

### Moving Beyond Black-Box Inference
* **Challenge**: Creating a responsive grid that allows users to input varying numbers of sources and destinations without breaking the layout.
* **Resolution**: Utilized Tkinter’s grid geometry manager to dynamically generate input fields, allowing for a flexible interface that scales up to a 4x4 logistics matrix.

---

## Visuals and Demos

### Application Interface
*Interactive GUI for solving transportation problems with real-time cost optimization.*
<img src="transportation.jpg" alt="Stepping Stone Solution Interface" width="100%"/>

### Application Walkthrough
*Demonstration of the VAM and Stepping Stone algorithms in action, from inputting supply/demand to finding the optimal transportation route.*

> [!IMPORTANT]
> **[Watch High-Resolution Transportation Optimizer Demo](transport.mp4)**

---

## Contact and Links
**Anurag Gaonkar** - [GitHub](https://github.com/AnuragGaonkar) | [LinkedIn](https://www.linkedin.com/in/anurag-gaonkar-68a463261)

Project Link: [https://github.com/AnuragGaonkar/STEPPING-STONE-SOLUTION](https://github.com/AnuragGaonkar/STEPPING-STONE-SOLUTION)
