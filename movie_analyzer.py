import torch
import dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.utils import clip_grad_norm_
# Load and preprocess the IMDb dataset
def load_and_preprocess_data():
    # Load your dataset and preprocess it
    # For simplicity, let's assume you have two lists: 'texts' and 'labels' where texts are the reviews and labels are their sentiment (0 for negative, 1 for positive)
    df = dataset.preprocessing()
    # Split the data into training and testing sets
    texts_train, texts_test, labels_train, labels_test = train_test_split(df['processed_review'], df['sentiment'], test_size=0.2, random_state=42)

    # Tokenize the texts using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_train = tokenizer(list(texts_train), padding=True, truncation=True, return_tensors='pt')
    tokens_test = tokenizer(list(texts_test), padding=True, truncation=True, return_tensors='pt')

    # Convert labels to numerical values
    label_mapping = {'positive': 1, 'negative': 0}
    labels_train = [label_mapping[label] for label in labels_train]
    labels_test = [label_mapping[label] for label in labels_test]
    # Convert labels to PyTorch tensors
    labels_train = torch.tensor(labels_train)
    labels_test = torch.tensor(labels_test)

    # Create PyTorch DataLoader for training and testing sets
    train_dataset = TensorDataset(tokens_train['input_ids'], tokens_train['attention_mask'], labels_train)
    test_dataset = TensorDataset(tokens_test['input_ids'], tokens_test['attention_mask'], labels_test)

    print("load and preprocess done")
    return train_dataset, test_dataset

# Define the BERT-based model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def train_model(train_loader, model, optimizer, criterion, epochs=3):
    model.train()
    print("model trained")
    for epoch in range(epochs):
        for inputs_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed
            optimizer.step()
        print(epoch)

def evaluate_model(test_loader, model):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs_ids, attention_mask, labels in test_loader:
            outputs = model(inputs_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

if __name__ == "__main__":
    train_dataset, test_dataset = load_and_preprocess_data()

    # Set batch size and create DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("dataloaded")
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    print("optimizer and loss")
    # Train the model
    train_model(train_loader, model, optimizer, criterion, epochs=3)
    print("trained")
    # Evaluate the model
    accuracy = evaluate_model(test_loader, model)
    print(f"Accuracy on the test set: {accuracy}")


