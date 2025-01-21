import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import random


# Set random seed
def set_seed(seed):
    """
    Set the seed for reproducibility.

    Parameters:
    - seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Model definition
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 2)  # Assuming binary classification

    def forward(self, x):
        return self.linear(x)


def preprocess_data(data: pd.DataFrame):
    """
    Preprocess the input data to:
    1. Fill missing values for numerical and categorical columns.
    2. Encode categorical columns.
    3. Convert the label column to numerical format.
    4. Standardize numerical features.

    Parameters:
    - data (pd.DataFrame): The input dataset.

    Returns:
    - X (pd.DataFrame): Processed features.
    - y (pd.Series): Processed label column.
    """
    label_column = "income"
    data = data.copy()

    # Fill missing values for all columns except the label column
    for column in data.columns:
        if column == label_column:
            continue  # Skip the label column
        if data[column].dtype == 'object':  # Categorical
            data[column] = data[column].fillna("missing")
        else:  # Numerical
            data[column] = data[column].fillna(data[column].median())

    # Encode categorical columns (excluding the label column)
    for column in data.select_dtypes(include=['object']).columns:
        if column != label_column:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])

    # Convert label column to numerical format
    if data[label_column].dtype == 'object':  # Label is string
        le = LabelEncoder()
        data[label_column] = le.fit_transform(data[label_column])
        print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Separate features and target
    X = data.drop(columns=[label_column])  # Features
    y = data[label_column]  # Target

    # Standardize numerical columns
    numeric_columns = X.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    return X, y


def train_model(optimizer_name, X_train_tensor, y_train_tensor, epochs=50):
    """
    Train the Logistic Regression model using a specified optimizer and batch size.

    Parameters:
    - optimizer_name (str): Name of the optimizer (e.g., "SGD", "Adam").
    - X_train_tensor (torch.Tensor): Features as PyTorch tensors.
    - y_train_tensor (torch.Tensor): Labels as PyTorch tensors.
    - batch_size (int): Batch size for training.
    - epochs (int): Number of training epochs.

    Returns:
    - model (nn.Module): Trained model.
    - losses (list): List of training loss values (per epoch).
    """
    batch_size = X_train_tensor.shape[0]
    input_dim = X_train_tensor.shape[1]
    model = LogisticRegressionModel(input_dim)
    criterion = nn.CrossEntropyLoss()

    lr = 0.01

    # Select optimizer based on the name
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "NAdam":
        optimizer = optim.NAdam(model.parameters(), lr=lr)
    elif optimizer_name == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=lr)
    elif optimizer_name == "Rprop":
        optimizer = optim.Rprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Create DataLoader for mini-batch training
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)  # Average loss for the epoch
        losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    return model, losses


def evaluate(model, X_test_tensor, y_test_tensor):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model (nn.Module): Trained model.
    - X_test_tensor (torch.Tensor): Test features.
    - y_test_tensor (torch.Tensor): Test labels.

    Returns:
    - accuracy (float): Accuracy on the test dataset.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, axis=1)
        accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
    return accuracy


def plot_training_loss_group(optimizer_losses, group, filename, group_colors):
    """
    Plot and save the training loss curves for a specific group of optimizers.

    Parameters:
    - optimizer_losses (dict): A dictionary where keys are optimizer names and values are tuples of
      (dirty_losses, repaired_losses).
    - group (list): A list of optimizer names to include in this plot.
    - filename (str): The filename for saving the plot.
    - group_colors (list): A list of colors corresponding to the optimizers in the group.
    """
    plt.figure(figsize=(8, 5))

    for idx, optimizer in enumerate(group):
        if optimizer in optimizer_losses:
            dirty_losses, repaired_losses = optimizer_losses[optimizer]
            color = group_colors[idx % len(group_colors)]  # Ensure unique color for each optimizer
            
            # Plot dirty losses (no marker)
            plt.plot(
                range(1, len(dirty_losses) + 1),
                dirty_losses,
                label=f"BaseVE + {optimizer}",
                linestyle='-',  # Solid line for dirty data
                lw=1,
                color=color
            )
            
            # Control marker density for repaired losses
            num_points = len(repaired_losses)
            num_markers = 10  # Number of markers to display
            marker_indices = np.linspace(0, num_points - 1, num_markers, dtype=int)

            # Plot repaired losses with sparse markers
            plt.plot(
                range(1, len(repaired_losses) + 1),
                repaired_losses,
                label=f"DE4ML + {optimizer}",
                linestyle='-',  # Solid line
                marker='*',  # Star marker for repaired data
                markevery=marker_indices,  # Only place markers at specified indices
                ms=15,  # Marker size
                lw=1,
                color=color
            )
    
    x_labels = ['0', '10', '20', '30', '40', '50']
    x = np.array([i * 10 for i in range(len(x_labels))])
    plt.xticks(x, x_labels)
    plt.xticks(rotation=20)
    plt.tick_params(labelsize=28)
    plt.ylabel("Training Loss", fontsize=28)
    plt.legend(loc="upper right", fontsize=14, ncol=2)  # Adjust legend size to fit all lines

    # Save the plot as a PDF file
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot for group saved to {filename}")


def train(data_paths, test_path, seed=None):
    """
    Train a model using data from both dirty and repaired datasets and evaluate on a common test dataset.

    Parameters:
    - data_paths (dict): Dictionary containing paths for dirty and repaired training datasets.
    - test_path (str): Path to the common test dataset.
    - seed (int): Random seed for reproducibility.

    Returns:
    - optimizer_losses (dict): A dictionary containing losses for all optimizers if SGD satisfies the condition.
    """
    if seed is not None:
        set_seed(seed)

    # Load and preprocess training data
    data_types = ["dirty", "repaired"]
    datasets = {}

    for data_type in data_types:
        data_path = data_paths[data_type]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")

        data = pd.read_csv(data_path)
        print(f"Loaded {data_type} training data preview:")
        # print(data.head())

        X, y = preprocess_data(data)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        datasets[data_type] = (X_tensor, y_tensor)

    # Load and preprocess test data
    test_data = pd.read_csv(test_path)
    X_test, y_test = preprocess_data(test_data)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Define optimizers
    optimizers = [
        "SGD", "Adam", "AdamW", "Adagrad", 
        "RMSprop", "NAdam", "ASGD", "Rprop"
    ]
    optimizer_losses = {}
    
    # Vary optimizers
    for optimizer in optimizers:
        print(f"Training with {optimizer} optimizer...")

        # Train models on dirty and repaired datasets
        model_dirty, dirty_losses = train_model(optimizer, *datasets["dirty"])
        model_repaired, repaired_losses = train_model(optimizer, *datasets["repaired"])

        optimizer_losses[optimizer] = (dirty_losses, repaired_losses)

        # Evaluate on the test dataset
        accuracy_dirty = evaluate(model_dirty, X_test_tensor, y_test_tensor)
        accuracy_repaired = evaluate(model_repaired, X_test_tensor, y_test_tensor)

        # Save evaluation results to a file
        with open(f"results/{optimizer}_evaluation_results.txt", "w") as f:
            f.write(f"Dirty Data Accuracy: {accuracy_dirty:.4f}\n")
            f.write(f"Repaired Data Accuracy: {accuracy_repaired:.4f}\n")

        print(f"Optimizer: {optimizer}, Dirty Data Accuracy: {accuracy_dirty:.4f}, Repaired Data Accuracy: {accuracy_repaired:.4f}")

    # Define optimizer groups
    group1 = ["Rprop", "AdamW", "Adagrad", "ASGD"]
    group2 = ["SGD", "Adam", "RMSprop", "NAdam"]

    # Define colors for each group
    group1_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
    group2_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]  # Purple, Brown, Pink, Gray

    # Plot each group
    plot_training_loss_group(optimizer_losses, group1, "results/fig_8a.pdf", group1_colors)
    plot_training_loss_group(optimizer_losses, group2, "results/fig_8b.pdf", group2_colors)

    return optimizer_losses


if __name__ == "__main__":
    # Define paths for dirty and repaired training datasets
    data_paths = {
        "dirty": "../../data/adult_balanced/train/adult_balanced_dirty.csv",
        "repaired": "../../data/adult_balanced/train/adult_balanced_repair.csv"
    }

    # Define path for common test dataset
    test_path = "../../data/adult_balanced/test/adult_balanced_clean.csv"

    seed = 4696
    train(data_paths, test_path, seed=seed)
