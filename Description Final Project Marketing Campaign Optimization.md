**Final Project**  
**Marketing Campaign Optimization** 

**Table of Contents**

[**1\. Project Objective	2**](#1.-project-objective)

[**2\. Dataset	2**](#2.-dataset)

[**3\. Business Scenario	3**](#3.-business-scenario)

[**4\. Problem Formulation	3**](#4.-problem-formulation)

[Part A: Causal Inference Analysis	3](#part-a:-causal-inference-analysis)

[Part B: Uplift Modeling	3](#part-b:-uplift-modeling)

[**5\. Common Pipeline	4**](#5.-common-pipeline)

[5.1 Data Preprocessing	4](#5.1-data-preprocessing)

[5.2 Exploratory Data Analysis	4](#5.2-exploratory-data-analysis)

[5.3 Causal Inference Analysis	5](#5.3-causal-inference-analysis)

[5.4 Uplift Modeling	5](#5.4-uplift-modeling)

[5.5 Campaign Targeting Policy	5](#5.5-campaign-targeting-policy)

[**6\. Causal Inference Analysis	6**](#6.-causal-inference-analysis)

[6.1 Treatment Assignment Analysis	6](#6.1-treatment-assignment-analysis)

[6.2 Causal Graph (DAG)	6](#6.2-causal-graph-\(dag\))

[6.3 Observable vs. Unobservable Confounding	6](#6.3-observable-vs.-unobservable-confounding)

[6.4 Estimator Comparison	7](#6.4-estimator-comparison)

[6.5 Business Interpretation	7](#6.5-business-interpretation)

[**7\. Evaluation	8**](#7.-evaluation)

[7.1 Causal Estimation	8](#7.1-causal-estimation)

[7.2 Uplift Modeling	8](#7.2-uplift-modeling)

[7.3 Policy Evaluation	8](#7.3-policy-evaluation)

[7.4 Ablation Study	8](#7.4-ablation-study)

[**8\. Engineering Requirements	9**](#8.-engineering-requirements)

[Suggested Project Structure	9](#suggested-project-structure)

[Technical Requirements	10](#technical-requirements)

[Code Quality	10](#code-quality)

[**9\. Expected Outcome	10**](#9.-expected-outcome)

# **1\. Project Objective** {#1.-project-objective}

The goal of this project is to build an end-to-end decision support system for **Marketing Campaign Optimization** using **Causal Inference** and **Uplift Modeling**.

Instead of sending promotional campaigns to every customer, the objective is to estimate:

* The average impact of a campaign (ATE)  
* The heterogeneous treatment effect for each customer (CATE)  
* Which customers should receive the campaign  
* Which customers should not receive any treatment  
* The expected business impact under different targeting policies

Students will implement a complete pipeline from raw data to causal analysis, uplift modeling, policy optimization, and business evaluation.

The project should include:

* Data preprocessing  
* Exploratory data analysis  
* Causal assumption analysis  
* Average Treatment Effect estimation  
* Individual Treatment Effect estimation  
* Uplift modeling  
* Campaign targeting policy  
* Offline evaluation  
* Business impact analysis

# **2\. Dataset** {#2.-dataset}

### **Dataset Used: Criteo Uplift Prediction Dataset ([https://ailab.criteo.com/criteo-uplift-prediction-dataset/](https://ailab.criteo.com/criteo-uplift-prediction-dataset/))**

The dataset contains customer-level information collected during a digital advertising campaign.

Each observation includes:

* Customer features  
* Treatment assignment  
* Conversion outcome

The dataset is designed for causal inference and uplift modeling research.

# **3\. Business Scenario** {#3.-business-scenario}

A marketing team plans to launch a promotional campaign.

However, contacting every customer is expensive and may even reduce customer engagement if promotions are sent to the wrong audience.

The business wants to answer the following questions:

* Does the campaign actually increase conversions?  
* What is the overall campaign effectiveness?  
* Which customers benefit from the campaign?  
* Which customers should not receive the campaign?  
* What targeting strategy maximizes business value?

The final objective is to maximize incremental conversions while minimizing campaign cost.

# **4\. Problem Formulation** {#4.-problem-formulation}

The project consists of two major components.

## **Part A: Causal Inference Analysis** {#part-a:-causal-inference-analysis}

Estimate the causal effect of the marketing campaign.

Students should estimate:

* Average Treatment Effect (ATE)  
* Conditional Average Treatment Effect (CATE)

Supported estimation methods include:

* Difference in Means  
* Regression Adjustment  
* Propensity Score Matching  
* Inverse Propensity Weighting (IPW)  
* Doubly Robust Estimation (DR)

Students should compare different estimators and discuss their assumptions.

## **Part B: Uplift Modeling** {#part-b:-uplift-modeling}

Build models to estimate individual treatment effects.

Supported approaches include:

### **Meta Learners**

* S-Learner  
* T-Learner  
* X-Learner  
* R-Learner  
* DR-Learner

### **Tree-based Methods**

* Causal Tree  
* Causal Forest  
* Uplift Tree

### **Representation-based Methods**

* DragonNet

The output should be:

* Estimated CATE  
* Customer uplift score  
* Customer ranking  
* Campaign recommendation

# **5\. Common Pipeline** {#5.-common-pipeline}

## **5.1 Data Preprocessing** {#5.1-data-preprocessing}

Perform data cleaning.

Examples include:

* Handle missing values   
* Remove invalid records  
* Normalize numerical features (if necessary)

## **5.2 Exploratory Data Analysis** {#5.2-exploratory-data-analysis}

Perform exploratory analysis to understand:

* Treatment ratio  
* Conversion rate  
* Feature distributions  
* Missing values  
* Feature correlations

Students should discuss potential biases observed in the dataset.

## **5.3 Causal Inference Analysis** {#5.3-causal-inference-analysis}

Estimate the overall treatment effect using multiple causal estimators.

Compare:

* Difference in Means  
* Regression Adjustment  
* IPW  
* Doubly Robust

Discuss:

* estimation differences  
* robustness  
* assumptions

## **5.4 Uplift Modeling** {#5.4-uplift-modeling}

Train one or more uplift models in Section 4\.

## **5.5 Campaign Targeting Policy** {#5.5-campaign-targeting-policy}

Design a marketing strategy based on estimated uplift.

Possible targeting strategies include:

* Treat everyone  
* Treat nobody  
* Random targeting  
* Treat customers with positive uplift  
* Treat Top 10% uplift customers  
* Treat Top 20% uplift customers  
* Treat Top-K customers maximizing expected business gain

Compare the effectiveness of different targeting policies.

# **6\. Causal Inference Analysis** {#6.-causal-inference-analysis}

Students must provide a complete causal analysis instead of only training machine learning models.

## **6.1 Treatment Assignment Analysis** {#6.1-treatment-assignment-analysis}

Investigate whether the treatment assignment appears randomized.

Possible analyses include:

* Treatment distribution  
* Covariate balance  
* Standardized Mean Difference (SMD)  
* Logistic regression predicting treatment  
* Propensity score distribution

Questions to discuss:

* Does treatment appear randomized ?  
* Can customer features predict treatment assignment ?  
* What evidence supports your conclusion ?

## **6.2 Causal Graph (DAG)** {#6.2-causal-graph-(dag)}

Construct a causal graph representing the business scenario.

The graph should identify:

* Treatment  
* Outcome  
* Potential confounders  
* Possible mediators  
* Possible colliders

Discuss whether the causal effect is identifiable.

## **6.3 Observable vs. Unobservable Confounding** {#6.3-observable-vs.-unobservable-confounding}

Assume two scenarios.

### **Scenario 1**

Treatment assignment is randomized.

Discuss:

* Why simple Difference in Means consistently estimates ATE.  
* Whether propensity score adjustment is still necessary.

### **Scenario 2**

Treatment assignment is observational.

Discuss:

* Possible confounders.  
* Why naive estimation becomes biased.  
* Which adjustment methods should be applied.  
* Which assumptions become necessary.

## **6.4 Estimator Comparison** {#6.4-estimator-comparison}

Estimate ATE using multiple causal estimators.

Compare:

* Estimated ATE  
* Bias and variance  
* Computational complexity  
* Practical advantages and disadvantages

Discuss when each estimator should be preferred.

## **6.5 Business Interpretation** {#6.5-business-interpretation}

Interpret the estimated treatment effect.

Examples:

* Is the campaign effective ?  
* Should the campaign continue ?  
* Which customer segments benefit the most ?  
* Which segments may be harmed by the treatment ?

Support conclusions using both numerical results and visualizations.

# **7\. Evaluation** {#7.-evaluation}

## **7.1 Causal Estimation** {#7.1-causal-estimation}

Evaluate:

* Estimated ATE  
* ATT (optional)  
* CATE distribution  
* Confidence intervals

## **7.2 Uplift Modeling** {#7.2-uplift-modeling}

Evaluate uplift performance using:

* Qini Curve  
* AUUC  
* Uplift Curve  
* Incremental Gain Curve  
* Calibration by Decile  
* Average Uplift

Students should compare multiple uplift models.

## **7.3 Policy Evaluation** {#7.3-policy-evaluation}

Compare different targeting policies.

Possible metrics include:

* Incremental conversions  
* Incremental gain  
* Campaign coverage

Discuss the trade-off between campaign size and business value.

## **7.4 Ablation Study** {#7.4-ablation-study}

Compare different estimation approaches.

Examples:

* Difference in Means vs IPW  
* IPW vs Doubly Robust  
* S-Learner vs T-Learner  
* T-Learner vs X-Learner  
* X-Learner vs DR-Learner  
* Meta Learners vs Causal Forest

Discuss why performance differs.

# **8\. Engineering Requirements** {#8.-engineering-requirements}

## **Suggested Project Structure** {#suggested-project-structure}

project/

├── data/

├── notebooks/

├── features/

├── causal/

│   ├── ate/

│   ├── propensity/

│   └── doubly\_robust/

├── uplift/

│   ├── meta\_learners/

│   ├── tree\_models/

│   └── deep\_models/

├── evaluation/

├── visualization/

├── report/

└── README.md

## **Technical Requirements** {#technical-requirements}

Recommended libraries:

* Python  
* Pandas  
* NumPy  
* Scikit-learn  
* EconML  
* CausalML  
* DoWhy (optional)  
* Matplotlib  
* XGBoost / LightGBM / CatBoost

## **Code Quality** {#code-quality}

The project should provide:

* Reproducible experiments  
* Modular implementation  
* Well-documented code  
* Clear visualization  
* GitHub repository with README

# **9\. Expected Outcome** {#9.-expected-outcome}

By completing this project, students will be able to:

* Formulate business problems using the potential outcomes framework.  
* Analyze treatment assignment mechanisms and evaluate whether causal identification assumptions are plausible.  
* Estimate Average Treatment Effects (ATE) using multiple causal inference methods.  
* Estimate Conditional Average Treatment Effects (CATE) with modern uplift modeling techniques.  
* Compare meta-learners, tree-based methods, and representation-based models for heterogeneous treatment effect estimation.  
* Design data-driven campaign targeting policies based on predicted uplift rather than response prediction alone.  
* Evaluate targeting strategies using uplift-specific metrics such as Qini Curve, AUUC, and Incremental Gain.  
* Translate causal inference results into actionable business recommendations for marketing campaign optimization.  
* Build an end-to-end causal analytics workflow that integrates data preprocessing, causal estimation, uplift modeling, policy optimization, and business evaluation.

