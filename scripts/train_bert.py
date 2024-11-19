import itertools
import torch
from transformers import BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


# Step 1: Load Tokenized Data

print("Loading tokenized data...")
train_tokens = torch.load('train_tokens.pt')
val_tokens = torch.load('val_tokens.pt')
test_tokens = torch.load('test_tokens.pt')

train_labels = torch.load('train_labels.pt')
val_labels = torch.load('val_labels.pt')
test_labels = torch.load('test_labels.pt')


# Step 2: Hyperparameter Grid

param_grid = {
    'learning_rate': [5e-5, 3e-5, 2e-5],
    'batch_size': [16, 32],
    'epochs': [3, 6],
}


# Step 3: Function to Compute Class Weights

def compute_weights(labels):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=torch.unique(labels).numpy(),
        y=labels.numpy()
    )
    return torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')


# Step 4: Train and Validate

def train_and_validate(params):
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']

    print(f"\nTraining with parameters: {params}")

    # Compute class weights
    class_weights = compute_weights(train_labels)

    # Create DataLoaders
    print("Preparing DataLoaders...")
    train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and optimizer
    print("Initializing BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_accuracy = 0
    patience = 2
    patience_counter = 0

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
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

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
            patience_counter = 0
            torch.save(model.state_dict(), f"bert_model_lr{learning_rate}_bs{batch_size}_epoch{epoch + 1}.pt")
            print(f"Model checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement.")
                break

    return best_val_accuracy


# Step 5: Hyperparameter Search

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


# Step 6: Evaluate on Test Set

def test_model(params):
    print("\nEvaluating best model on test set...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.load_state_dict(torch.load(f"bert_model_lr{params['learning_rate']}_bs{params['batch_size']}_epoch{params['epochs']}.pt"))
    model.eval()

    test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_labels)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

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

