import coremltools as ct

# Load the model
model = ct.models.MLModel('models/sentiment_model_metal_metal.mlpackage')

# Print model input and output descriptions
print("Model Inputs:")
print(model.input_description)
print("\nModel Outputs:")
print(model.output_description)

# Print more detailed model specification
print("\nDetailed Model Specification:")
print(model.get_spec())
