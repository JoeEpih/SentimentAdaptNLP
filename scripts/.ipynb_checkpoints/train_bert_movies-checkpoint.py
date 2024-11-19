import itertools
import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


# Step 1: Load Tokenized Movies Data

print("Loading pre-tokenized Movies dataset...")
test_tokens = torch.load('test_tokens.pt')  # Pre-tokenized entire dataset
test_labels = torch.load('test_labels.pt')  # Corresponding labels


# Step 2: Split the Tokenized Dataset

print("Splitting the tokenized dataset...")
dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_labels)

# Split into 80% training and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# Step 3: Hyperparameter Grid

param_grid = {
    'learning_rate': [5e-5, 3e-5, 2e-5],
    'batch_size': [16, 32],
    'epochs': [3, 6]
}


# Step 4: Function to Compute Class Weights

def compute_weights(labels):
    train_labels_split = [sample[2].item() for sample in train_dataset]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=torch.unique(labels).numpy(),
        y=train_labels_split
    )
    return torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')


# Step 5: Train and Validate

def train_and_validate(params):
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']

    print(f"\nTraining with parameters: {params}")

    # Compute class weights
    class_weights = compute_weights(test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and optimizer
    print("Initializing BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_accuracy = 0

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Save model if validation improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"bert_movies_lr{learning_rate}_bs{batch_size}_epoch{epoch + 1}.pt")
            print(f"Model checkpoint saved.")

    return best_val_accuracy


# Step 6: Hyperparameter Search

best_params = None
best_accuracy = 0

for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    val_accuracy = train_and_validate(param_dict)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = param_dict

print(f"\nBest parameters: {best_params}")
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")


# Step 7: Evaluate on Test Set

def test_model(params):
    print("\nEvaluating best model on test set...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.load_state_dict(torch.load(f"bert_movies_lr{params['learning_rate']}_bs{params['batch_size']}_epoch{params['epochs']}.pt"))
    model.eval()

    test_loader = DataLoader(dataset, batch_size=params['batch_size'])

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

test_model(best_params)

