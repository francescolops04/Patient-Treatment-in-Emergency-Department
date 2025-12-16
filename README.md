# Patient-Treatment-in-Emergency-Department

## 📋 Overview

This project performs an advanced **Process Mining analysis** on a real-world dataset comprising **25,115 clinical records** from an Emergency Department (ED). By leveraging the **PM4Py** framework, the analysis transforms raw, noisy hospital logs into actionable strategic insights.

The goal is to move beyond simple descriptive statistics to identify **structural bottlenecks**, **resource imbalances** ("Hero Culture"), and **non-stationary behaviors** that jeopardize patient safety and operational efficiency.

## 🚀 Key Insights & Findings

Based on the analysis of **1,801 valid end-to-end patient traces**:

  * **⚠️ The "Hero Culture" Risk:** Resource allocation is critically unbalanced. **Resource 1.0** handles \~4x the workload of others (CV: **2.82**), representing a dangerous Single Point of Failure.
  * **📉 System Instability:** The system fails **Little's Law** with a **79.6% discrepancy** between real and theoretical WIP, indicating chronic congestion and "bursty" arrival waves.
  * **⏳ The "Bow-Tie" Bottleneck:** The process starts and ends linearly but descends into chaos during the **Vital Sign Check** phase (Median wait: **0.58 hours**), characterized by intensive self-loops.
  * **🏥 Clinical vs. Process Delays:** Using **Data-Aware Clustering**, we discovered that **50% of "Routine" patients** suffer long delays without clinical justification, isolating administrative inefficiency.

## 🛠️ The Pipeline

The `emergency_department_analysis.py` script executes a rigourous **Knowledge Uplift Trail**:

1.  **Ingestion & Semantic Mapping:** Consolidating fragmented nurse IDs into a single `org:resource` attribute.
2.  **Clinical Cleaning:** Filtering biologically impossible values (e.g., HR \> 250 bpm, Temperature \> 110°F).
3.  **Trace Selection:** Removing "Zombie Cases" (\> 32.4 hours) and "Censored Traces" to stabilize the log.
4.  **Process Discovery:** Generating Petri Nets (Inductive/Heuristic) and Directly Follows Graphs (DFG).
5.  **Statistical Validation:**
      * **Chi-Square ($p < 10^{-60}$):** Proves Triage effectively predicts outcome.
      * **ANOVA ($p < 0.001$):** Confirms heart rate correlates with acuity.
6.  **Optimization:** Automatically finding the best `K=3` variants to balance Fitness (0.99) and Precision.

## 💻 Getting Started

### Prerequisites

You need **Python 3.x** installed. The core analysis relies heavily on the `pm4py` library.

### Installation

Clone the repository and install the required dependencies. **Note:** You must install `pm4py` separately if you haven't already.

```bash
# 1. Install standard data science libraries
pip install pandas matplotlib seaborn numpy scipy

# 2. Install Process Mining framework
pip install pm4py
```

### Usage

Place your dataset file (e.g., `dataset_for_exam.csv`) in the root directory and run the script:

```bash
python emergency_department_analysis.py
```

## 📊 Visualizations

The script generates several critical plots to visualize the process "health":
| Category | Chart Type | Analytical Purpose |
| :--- | :--- | :--- |
| **Process Models** | **DFG (Bow-Tie)** | Visualizes the chaotic flow and self-loops between Vital Signs and Medication. |
| | **Petri Nets** | Sound models (Inductive Miner) vs. Main Flow models (Heuristic Miner). |
| **Temporal Analysis** | **Histogram + KDE** | Analysis of Length of Stay (LoS) distribution (Skewness). |
| | **Boxplots** | Lead Time stratification by **Acuity Level** (Critical vs Routine). |
| **Bottlenecks** | **Boxplots** | Waiting times distribution for each specific activity. |
| | **Bar Chart** | **Arrival Rate** by hour (Seasonality & Intake Peaks). |
| **Stability** | **Line Chart** | **Real vs. Theoretical WIP** evolution (Little's Law verification). |
| | **Histogram + KDE** | **Inter-Arrival Times (IAT)** to detect Burstiness patterns. |
| **Variants** | **Pareto Chart (Line)** | Cumulative coverage of variants (The "Long Tail" problem). |
| | **Violin Plots** | Detailed performance comparison of the Top-10 variants. |
| **Optimization** | **Bar Chart** | Patient distribution by **Data-Aware Patterns** (Vectors). |
| | **Line Chart** | Fitness/Precision/F1 Score trade-off to select optimal `K`. |

## 🧪 Methodological Highlights

  * **Data-Aware Mining:** We encoded cases into binary vectors $V = (P1, P2, P3)$ based on Criticality, Instability, and Efficiency to cluster patients significantly better than standard grouping.
  * **Reliability:** Removed 8,289 "dirty" records to prevent artificial inflation of discovery algorithms.

*Analysis performed using Python & PM4Py.*
