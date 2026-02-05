import pandas as pd

def predict_cost(pipeline, shipment):
    """runs a shipment record through the pipeline and returns an estimated cost

    Args:
        pipeline: trained Pipeline used to predict the cost
        shipment: single shipment record with details for prediction

    returns:
    float: estimated shipment cost
        """
    df = pd.DataFrame([shipment])
    prediction = pipeline.predict(df)
    return prediction[0]

