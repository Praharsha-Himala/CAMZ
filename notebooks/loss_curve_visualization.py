import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def plot_loss_curves(file_path):
#visualizing loss curves
    data = pd.read_csv(file_path)

    print(f'The best loss of {data["loss"].min()} achieved at Epoch: {data.loc[data["loss"].idxmin(), "epoch"]}')


    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(data['epoch'], data['loss'], label='Training Loss', color='blue')

    # Plot validation loss
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', color='red')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot
    plt.grid()
    plt.show()

def plot_training_history(batch_train_path, batch_val_path, epoch_history_path):
    """
    Visualizes batch-wise and epoch-wise training history in a single plot.
    
    Parameters:
    batch_train_path (str): Path to batch-wise training loss CSV file.
    batch_val_path (str): Path to batch-wise validation loss CSV file.
    epoch_history_path (str): Path to epoch-wise training and validation loss CSV file.
    """
    # Load data
    batch_train_loss = pd.read_csv(batch_train_path)
    batch_val_loss = pd.read_csv(batch_val_path)
    epoch_loss = pd.read_csv(epoch_history_path)
    
    # Assign batch indices
    batch_train_loss['batch'] = range(len(batch_train_loss))
    batch_val_loss['batch'] = np.linspace(0, len(batch_train_loss), len(batch_val_loss))
    epoch_loss['epoch'] = np.linspace(0, len(batch_train_loss), len(epoch_loss))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(batch_train_loss['batch'], batch_train_loss['loss'], label='Batch-wise Training Loss', alpha=0.7)
    plt.plot(batch_val_loss['batch'], batch_val_loss['loss'], label='Batch-wise Validation Loss', alpha=0.7)
    plt.plot(epoch_loss['epoch'], epoch_loss['loss'], label='Epoch-wise Training Loss', color='darkblue')
    plt.plot(epoch_loss['epoch'], epoch_loss['val_loss'], label='Epoch-wise Validation Loss', color='darkred')
    
    # Labels and formatting
    plt.xlabel('steps')
    plt.ylabel('CIoU Loss')
    # plt.title('Training and Validation Loss History')?
    plt.legend()
    plt.grid()
    plt.show()



def Initialize_writer(file_path,columns = ['epoch','loss','val_loss']):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode='w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)