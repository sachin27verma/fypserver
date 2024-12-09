from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import unicodedata
import pandas as pd
import re
# import unicodedata // sksachin

# Initialize the Flask app
app = Flask(__name__)

# Load the model and tokenizer
MODEL_PATH = './fake_news_model'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set model to evaluation mode

# Define a dataset class for processing the input text
class NewsTestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts.reset_index(drop=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = tokenizer(self.texts[idx],
                         padding='max_length',
                         truncation=True,
                         max_length=512,
                         return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in item.items()}  # Remove batch dimension
        return item

# Preprocess the text

def preprocess_text(text):
    # Normalize the text
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()

    # Replace unwanted characters
    text = text.replace('\n', ' ').replace('\r', '').replace('\xa0', ' ')
    text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    text = text.replace('â€“', '-').replace('â€"', '"').replace('â€¦', '...')

    # Use regex to remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Clean up extra spaces and return
    text = text.replace('"', '').replace("'", '')
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space

    return text


# Prediction function
def predict_news(text):
    # Preprocess the text
    text = preprocess_text(text)

    # Prepare the input for the model
    test_data = pd.DataFrame({'text': [text]})
    test_data['text'] = test_data['text'].astype(str)

    # Create the dataset and dataloader
    test_dataset = NewsTestDataset(test_data['text'])
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.eval()
    predictions = []

    # Make predictions
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(model.device) for key, val in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            predictions.append(predicted_class)

    # Map predicted class to label
    label_map = {0: 'FAKE', 1: 'REAL'}
    return label_map[predictions[0]]

# Home endpoint
@app.route('/')
def home():
    return 'Fake news detection server is running!'

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json(force=True)
        
        if not isinstance(data, dict) or 'text' not in data:
            return jsonify({'error': 'Invalid input. JSON must contain a "text" field.'}), 400
        
        text = data['text']
        
        # Ensure text is a string
        if not isinstance(text, str):
            return jsonify({'error': '"text" field must be a string.'}), 400
        
        # Predict the label
        prediction = predict_news(text)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
