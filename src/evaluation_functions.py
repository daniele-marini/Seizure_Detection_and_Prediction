import torch

def get_correct_test_samples(scores: torch.Tensor, labels: torch.Tensor, current_second):
    """
    Get the number of correctly classified examples and errors per category.

    Args:
        scores: the probability distribution.
        labels: the class labels.

    Returns:
        correct: Number of correct samples.
        errors_per_category: Dictionary with the count of errors per category.
    """
    classes_predicted = torch.argmax(scores, 1)
    correct = (classes_predicted == labels).sum().item()

    false_positive = 0
    time=0
    errors_per_category = {}
    for label, prediction in zip(labels, classes_predicted):
      # calculate prediction time before the seizure
      if label == prediction and label.item()==1:
        time = current_second

      if label != prediction:
        # calculate the false positive pre-ictal prediction
        if prediction.item() == 1:
          false_positive+=1
        ## calculate false predictions for each class
        if label.item() in errors_per_category:
            errors_per_category[label.item()] += 1
        else:
            errors_per_category[label.item()] = 1

    return correct, errors_per_category, false_positive, time

def evaluate(model, test_dataloader, device):
    '''
    Evaluate the metrics needed for the task

    Args:
    - model : The trained neural network model
    - dataloader : A PyTorch DataLoader object that returns batches from the dataset
    - device : The device to run the model on (CPU or GPU)

    Returns:
    - the accuracy of the model
    - the summary of the errors for each class
    - the false positive predictions
    - the second of the first correct prediction
    '''

    correct = 0
    samples_val = 0
    errors_summary = {}
    FP = 0
    time_before = []

    model = model.eval()
    with torch.no_grad():
        for idx_batch, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            scores = model(images)
            samples_val += len(images)
            current_second = idx_batch
            correct_batch, errors_batch, false_positive, time = get_correct_test_samples(scores, labels, current_second)
            correct += correct_batch
            FP+=false_positive
            time_before.append(time)

            for label, error_count in errors_batch.items():
                if label in errors_summary:
                    errors_summary[label] += error_count
                else:
                    errors_summary[label] = error_count

    accuracy = 100. * correct / samples_val
    time_before = list(filter(lambda x: x != 0, time_before))
    time_before = time_before[0] if time_before!=[] else 0
    return accuracy, errors_summary, FP, time_before