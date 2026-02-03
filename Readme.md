#  Edge Offloading Decision System Using Machine Learning

## Project Description

This project addresses a complex engineering problem where edge devices (like smartphones, IoT nodes) must decide whether to execute a task **locally** or **offload it to the cloud**. 

The decision depends on multiple system parameters such as CPU usage, battery level, network latency, bandwidth availability and task complexity. A **machine learning model** is trained to predict the optimal offloading decision.

---

# Preview
![alt text](/Pics/image_4.PNG)

![alt text](/Pics/image_1.png)

![alt text](/Pics/image_2.png)

---

## Problem Statement

Modern edge devices have limited resources, yet they need to run heavy tasks in real-time. Offloading these tasks to the cloud or edge server can help, but it must be done **intelligently** based on:
- CPU usage
- Battery level
- Network conditions
- Task complexity
- Bandwidth Availability

The goal is to:
- Create a **ML model** that predicts:
  - `0`: Execute locally
  - `1`: Offload to cloud
- Build a **Python GUI** (Tkinter) that allows user input and shows real-time decisions

---

## Project Structure

Virtual environemnt shold be created within this directory. Python file names as **Decision Task.py** should be run within the directory. Overall no changes require in structure of project.

---

## Dataset

The dataset contains **10,000 rows**, unequally split between:
- `Decision = 0`: Local execution
- `Decision = 1`: Offload to cloud

It includes noisy samples (~5%) that reflects real-world imperfections.

### Features:
| Feature               | Description |
|-----------------------|-------------|
| `CPU_Usage`           | % CPU usage of device |
| `Battery_Level`       | % battery remaining |
| `Network_Latency`     | In milliseconds |
| `Bandwidth_Availability` | Network bandwidth in kbps |
| `Data_Size`           | Task data size in KB |
| `Task_Complexity`     | low / medium / high |
| `Decision`            | 0 = Local, 1 = Offload |

---

## Model Training

ML models tested:
- Random Forest (best performer)
- Decision Tree
- Logistic Regression
- K-Nearest Neighbor
- Support Vector Classifier
- MLP (Neural Network)

### Preprocessing:
- Label Encoding encoding for `Task_Complexity`
- StandardScaler applied to numeric features
- Train/test split: 80/20

### Evaluation:
- Accuracy
- Confusion Matrix
- F1-Score, Precision, Recall
- Feature Importance

### Result:
- Performance of each ML models with respect to the metrics were all reported in a file named as `Performance Result.md`.
- Bargraphs, Line graph were generated using **matplot lib**.

---
## Top Model Performance:
![alt text](/Pics/image_1.png)

![alt text](/Pics/image_2.png)

![alt text](/Pics/image_3.png)



---

## GUI Simulation (Tkinter)

### Features:
- User inputs real-time values for each feature
- Dropdown to select task complexity
- Button to predict whether to offload or execute locally
- Displays result in a popup window


### How to Use This Project
- Clone/download the repository
- Install dependencies:
    - pip install matplotlib seaborn numpy pandas scikit-learn imbalanced-learn

- To run the project:
    - Your terminal must be inside the directory.
    - Then run file: **Decision Task.py**




