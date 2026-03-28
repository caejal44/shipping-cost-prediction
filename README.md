# Shipping Cost Predictor
A machine learning–powered decision-support tool that helps users estimate and compare shipping costs across carriers.

Rather than returning a single prediction, this application evaluates multiple carriers and provides **cost, transit, and data-backed insights** to support real-world shipping decisions.

---

## Why This Matters

Shipping decisions in real-world logistics are not based on a single price quote.

This tool demonstrates how machine learning can be combined with historical performance data to:

- Compare multiple carrier options  
- Evaluate delivery risk  
- Provide transparency into data reliability  

The goal is to move from **prediction → decision support**.

---

## Application Overview

The app allows users to input shipment details and receive ranked carrier recommendations.

### Key interface components:

- Input panel for shipment details (origin, destination, weight, service, delivery date)  
- Ranked carrier comparison table  
- Supporting analytics across three dimensions:
  - Transit Time  
  - Cost Benchmark  
  - Data Coverage  

### From the UI:

- Carrier rankings include:
  - Estimated cost  
  - Average transit time  
  - Delivery risk indicator  

- Supporting charts provide context for decision-making  

---

## Key Features

### 1. Multi-Carrier Cost Prediction
Predicts shipping cost for all carriers servicing a given route using a trained ML model.

### 2. Carrier Ranking System
Carriers are ranked by estimated cost, with additional context:

- Average transit days  
- Delivery risk vs user-selected delivery date  

### 3. Transit Time Analysis
Visual comparison of carrier speed against the delivery target.

### 4. Cost Benchmarking
Compares predicted cost against historical averages within weight ranges.

### 5. Data Coverage Insight
Displays shipment volume per carrier to indicate reliability of estimates.

### 6. Real-Time Decision Support
Combines prediction + historical data to answer:

> “Which carrier is cheapest, and will it likely meet my delivery target?”

---

## How It Works

### Input

Users provide:

- Origin warehouse  
- Destination store  
- Shipment weight  
- Service level  
- Desired delivery date  

---

### Processing Pipeline

1. **Filter eligible carriers**  
   Based on historical route + service coverage  

2. **Generate predictions**  
   Each carrier is evaluated using the trained ML pipeline  

3. **Enhance with historical context**  
   - Average transit days  
   - Average cost (weight-bucketed)  
   - Shipment volume  

4. **Rank carriers**  
   - Sorted by estimated cost  
   - Annotated with delivery risk  

---

## Architecture

### Frontend
- Streamlit (interactive UI)

### Backend
- Python  
- Pandas / NumPy  
- Matplotlib  

### Machine Learning
- Scikit-learn pipeline  
- Model persisted via joblib  

---

## Project Structure

```bash
project_root/
├── app/
│   └── streamlit_app.py
├── model/
│   ├── training.py
│   ├── predicting.py
│   └── shipping_cost_pipeline.pkl
├── data/
│   ├── data_access.py
│   └── datasets/
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/caejal44/shipping-cost-prediction.git
cd shipping-cost-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app/streamlit_app.py
```

### 4. Open in browser

```bash
http://localhost:8501
```

---

## Model Details

- **Model Type:** Random Forest Regressor  
- **Framework:** Scikit-learn Pipeline  

### Features
- Origin  
- Destination  
- Carrier  
- Weight  
- Service  
- Distance  

### Target
- Shipping Cost  

---

### Training Behavior

- Model auto-trains if no saved pipeline exists  
- Persisted using `joblib`

## Model Note

The trained model file is not included in the repository due to size.

The application will automatically train the model on first run if no saved pipeline is found.  

---

## Data Design

This project uses a **simulated logistics dataset** designed to mimic real-world shipping behavior.

Includes:

- Lane-based routing (warehouse → store)  
- Carrier coverage by lane  
- Service-level constraints  
- Distance-based pricing logic  
- Weight-tier cost variation  

---

## Assumptions & Limitations

- Data is simulated, not real carrier pricing  
- Predictions are estimates, not quotes  
- Transit times are historical averages  

Does not account for:

- Real-time capacity constraints  
- Weather or disruptions  
- Dynamic pricing contracts  

---

## Future Improvements

- Zone-based pricing  
- Per-carrier model tuning  
- Export functionality (CSV / PDF)  
- Batch prediction input  
- Authentication layer  

---

## Project Motivation

This project evolved from a machine learning capstone into a **production-style portfolio application**.

**Goal:**

> Demonstrate how ML can be integrated into a business decision workflow, not just a standalone model.

---

## Author

**Joshua Lane**  
Software Engineering & Machine Learning Portfolio Project