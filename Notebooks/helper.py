import torch
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

def return_label_maps_from_adata(adata, category):
    label_to_id = {label: idx for idx, label in enumerate(adata.obs[category].unique())}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return(label_to_id, id_to_label)

def return_dataset_sizes(adata, ratio, pct_of_data=1):
    obs_num = round(adata.shape[0]*pct_of_data)
    train_obs_num = round(obs_num*ratio)
    test_obs_num = obs_num - train_obs_num
    return(train_obs_num, test_obs_num)

def obs_to_tensor(adata, category=None, training_size=None, testing_size=None,):
    
    def tensors_to_dataset(obs_list):
        tensors = []
        for obs in obs_list:
            tensors.append(
                torch.tensor(adata[adata.obs_names == obs].X.toarray())
            )
        tensors = torch.squeeze(torch.stack(tensors))
        if category != None:
            labels = []
            for obs in obs_list:
                labels.append(
                    adata[adata.obs_names == obs]
                    .obs[category]
                    .iloc[0]
                )
            labels = torch.tensor([label_to_id[label] for label in labels])
            return TensorDataset(tensors, labels)
        else:
            return tensors

    if testing_size + training_size <= adata.shape[0]:
        random_obs = np.random.choice(
            adata.obs.index,
            size=training_size + testing_size,
            replace=False,
        )
        random_obs_train = random_obs[:training_size]
        random_obs_test = random_obs[-testing_size:]
        if category != None:
            label_to_id, id_to_label = return_label_maps_from_adata(adata, category)
            training_data = tensors_to_dataset(random_obs_train)
            testing_data = tensors_to_dataset(random_obs_test)
            return (training_data, testing_data, id_to_label)
        else:
            training_data = tensors_to_dataset(random_obs_train)
            testing_data = tensors_to_dataset(random_obs_test)
            return (training_data, testing_data)

def evaluate_model(model, dataset, labels_map):

    def outputs_to_labels(output, labels_map):
        predicted_index = torch.argmax(output).item()
        predicted_label = labels_map[predicted_index]
        return predicted_label

    correct_guesses = []
    for data, label in dataset:
        model_output = model(data)
        predicted_label = outputs_to_labels(model_output, labels_map)
        actual_label = labels_map[int(label)]
        if predicted_label == actual_label:
            correct_guesses.append(predicted_label)
        else:
            continue
    
    print(f'The models accuracy is {(len(correct_guesses)/len(dataset))*100}%')
    print(f'The accuracy of random classification would be {(1/len(labels_map)*100)}%')


def plot_scaled_losses(train_losses, val_losses, num_epochs, log_scale=False):
    epochs = range(num_epochs)

    fig, ax1 = plt.subplots()

    ax1.plot(epochs, train_losses, 'b', label='Training loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='b')

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_losses, 'r', label='Validation loss')
    ax2.set_ylabel('Validation Loss', color='r')

    if log_scale == True:
        ax1.set_yscale('log')
        ax2.set_yscale('log')

    fig.tight_layout()
    plt.show()

def return_value_range(data_loader):
    max_vals = []
    min_vals = []
    for i in data_loader:
        max_vals.append(torch.max(i))
        min_vals.append(torch.min(i))
    return(int(min(min_vals)), int(max(max_vals)))