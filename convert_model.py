from core.models.converter import convert_and_optimize_model

convert_and_optimize_model(
    model_name="distilbert-base-uncased",
    output_path="/Users/cpconnor/CascadeProjects/llm-engine/models/sentiment_model_metal.mlpackage"
)
