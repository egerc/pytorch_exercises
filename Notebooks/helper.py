import torch
from torch.utils.data import TensorDataset
import numpy as np

def return_label_maps_from_adata(adata, category):
    label_to_id = {label: idx for idx, label in enumerate(adata.obs[category].unique())}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return(label_to_id, id_to_label)

def return_dataset_sizes(adata, ratio, pct_of_data=1):
    obs_num = round(adata.shape[0]*pct_of_data)
    train_obs_num = round(obs_num*ratio)
    test_obs_num = obs_num - train_obs_num
    return(train_obs_num, test_obs_num)

# This function takes in your adata object as well as the category you want to train your data on as well as parameters for defining the datasets sizes and returns two pytorch datasets, one for training and one for testing
def obs_to_tensor(adata, category=None, training_size=None, test_size=None):
    # This helper function returns a single pytorch dataset from the adata object given a list of observations
    def tensors_to_dataset(adata, obs_list, category, label_to_id):
        tensors = []
        labels = []
        for obs in obs_list:
            tensors.append(
                torch.tensor(
                    adata[adata.obs_names == obs].X
                    .toarray()
                )
            )
            labels.append(
                adata[adata.obs_names == obs].obs[category].iloc[0]
            )
        tensors = torch.squeeze(torch.stack(tensors))
        labels = torch.tensor([label_to_id[label] for label in labels])
        return TensorDataset(tensors, labels)

    # The purpose of these dictionaries is to encode the values of the desired category as integers
    label_to_id, id_to_label = return_label_maps_from_adata(adata, category)

    # Making sure the total dataset size doesnt exceed the number of observations in the adata object
    if test_size + training_size <= adata.shape[0]:
        # Randomly sampling from the observations of the adata object
        random_obs = np.random.choice(
            adata.obs.index,
            size=training_size + test_size,
            replace=False,
        )
        # Creating two subsets of the sampled observations for training and testing purposes
        random_obs_train = random_obs[:training_size]
        random_obs_test = random_obs[-test_size:]

        # Creating the datasets using the helper function from the two subset samples
        training_data = tensors_to_dataset(adata, random_obs_train, category, label_to_id)
        testing_data = tensors_to_dataset(adata, random_obs_test, category, label_to_id)

    return(
        training_data,
        testing_data,
        id_to_label,
    )


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