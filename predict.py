import torch
import numpy as np
import random
import gzip
import pickle
from model import RNNModel
from utils import preprocess  # Import preprocessing function

# ✅ Load the same dataset used for training
pkl_file_path = "./data/icentia11k.pkl"  # Update with actual path

# ✅ Load dataset from compressed file
with gzip.open(pkl_file_path, "rb") as f:
    x_data, labels = pickle.load(f)  # Load data (same as ECG_DataModule)

# ✅ Preprocess the data (same method used in ECG_DataModule)
x_data = preprocess(x_data)  # Normalize input
x_data = np.expand_dims(x_data, axis=(1, 2))  # Ensure correct shape (batch, 1, 1, input_size)

# ✅ Select 20 random samples from test data
sample_indices = random.sample(range(len(x_data)), 20)
test_samples = x_data[sample_indices]

print(f"✅ Selected {len(test_samples)} test samples for prediction.")

# ✅ Initialize model and load trained weights
torch.serialization.add_safe_globals([RNNModel])  # Allowlist model
model = RNNModel()
model.load_state_dict(torch.load("./results/model_last.pt", map_location="cpu", weights_only=False))

model.eval()

# ✅ Function to predict for multiple samples
def batch_predict(model, test_samples):
    predictions = []
    
    for sample in test_samples:
        sample = torch.tensor(sample, dtype=torch.float32)  # Convert to tensor
        hidden = model.initHidden(1)  # Initialize hidden state

        with torch.no_grad():
            output, _ = model(sample, hidden)  # Forward pass

        probabilities = torch.softmax(output, dim=1)  # Get class probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Get predicted label
        predictions.append((predicted_class, probabilities.numpy()))

    return predictions

# ✅ Run predictions on 20 test samples
predictions = batch_predict(model, test_samples)

# ✅ Print results
LABEL_MAPPING = {0: "Normal", 1: "PAC", 2: "PVC"}
for i, (pred_class, prob) in enumerate(predictions):
    print(f"Sample {i+1}: Predicted Class = {LABEL_MAPPING[pred_class]}, Probabilities = {prob}")
