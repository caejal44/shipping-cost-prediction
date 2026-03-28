# shipping-cost-prediction
A machine learning–powered decision-support tool that helps users estimate and compare shipping costs across carriers for a given route, service level, and shipment profile.
This application combines:
- ML-based cost estimation
- Historical transit performance
- Cost benchmarking
- Data coverage insights
The result is a business-oriented tool designed to support carrier selection decisions.

## Application Overview
The app allows users to input shipment details to receive ranked carrier recommendations.
Example interface:
- Input panel for shipment details (origin, destination, weight, service, delivery date)
- Ranked carrier comparison table
- Supporting analytics across three dimensions:
  - Transit Time
  - Cost Benchmark
  - Data Coverage

From the UI:
Carrier rankings include:
- Estimated cost
- Average transit time
- Delivery risk indicator
- Supporting charts provide context for decision-making

## Key Features

1)  Multi-Carrier Cost Prediction  
  Predicts shipping cost for all carriers servicing a given route using a trained ML model.
2) Carrier Ranking System  
  Carriers are ranked by estimated cost, with additional context:
    - Average transit days
    - Delivery risk vs user-selected delivery date
3) Transit Time Analysis  
  Visual comparison of carrier speed against the delivery target
4) Cost Benchmarking  
    Compares predicted cost against historical averages within weight ranges
5) Data Coverage Insight  
    Displays shipment volume per carrier to indicate reliability of estimates
6) Real-Time Decision Support   
    Combines prediction + historical data to answer:  
    “Which carrier is cheapest, and will it likely meet my delivery target?”

## How It Works
### Input
Users provide:
- Origin warehouse
- Destination store
- Shipment weight
- Service level
- Desired delivery date
### Processing Pipeline
1. Filter eligible carriers
   - Based on historical route + service coverage
2. Generate predictions
   - Each carrier is evaluated using the trained ML pipeline
3. Enhance with historical context
   - Average transit days
   - Average cost (weight-bucketed)
   - Shipment volume
5. Rank carriers
   - Sorted by estimated cost
   - Annotated with delivery risk

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
### Project Structure (Simplified)
project_root/  
  &nbsp;├--  app/  
  &nbsp;&nbsp;&nbsp;&nbsp;└--  streamlit_app.py  
  &nbsp;├--  model/  
  &nbsp;&nbsp;&nbsp;&nbsp;├--  training.py  
  &nbsp;&nbsp;&nbsp;&nbsp;├--  predicting.py  
  &nbsp;&nbsp;&nbsp;&nbsp;└--  shipping_cost_pipeline.pkl  
  &nbsp;├--  data/  
  &nbsp;&nbsp;&nbsp;&nbsp;├--  data_access.py  
  &nbsp;&nbsp;&nbsp;&nbsp;└--  datasets/  
  &nbsp;└--  README.md  
## Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/shipping-cost-predictor.git
cd shipping-cost-predictor
2. Install dependencies
pip install -r requirements.txt
3. Run the application
streamlit run app/streamlit_app.py
4. Open in browser
http://localhost:8501
🤖 Model Details
Model Type: Random Forest Regressor
Framework: Scikit-learn Pipeline
Features:
Origin
Destination
Carrier
Weight
Service
Distance
Target:
Shipping Cost
Training Behavior
Model auto-trains if no saved pipeline exists
Persisted using joblib
📊 Data Design

This project uses a simulated logistics dataset designed to mimic real-world shipping behavior.

Includes:

Lane-based routing (warehouse → store)
Carrier coverage by lane
Service-level constraints
Distance-based pricing logic
Weight-tier cost variation
⚠️ Assumptions & Limitations
Data is simulated, not real carrier pricing
Predictions are estimates, not quotes
Transit times are historical averages
Does not account for:
Real-time capacity constraints
Weather or disruptions
Dynamic pricing contracts
Future Improvements
Real-time pricing API integration
Enhanced feature engineering (seasonality, demand)
Per-carrier model tuning
Export functionality (CSV / PDF)
Deployment (AWS / Streamlit Cloud)
Authentication layer
Project Motivation

This project evolved from a machine learning capstone into a production-style portfolio application.

Goal:

Demonstrate how ML can be integrated into a business decision workflow, not just a standalone model.

Author

Joshua Lane
Software Engineering / Machine Learning Portfolio Project

