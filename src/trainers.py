import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import torch


class Dataset_from_sentences:
    """
    This class produces the input for the model
    Parameters:
        - sentences: list of sentences
        - cat_lists: lists of labels per category
    get_item sample:
        - ids (np.array): token's ids array
        - class"i"_target (np.array): tensor of targets for category i
    """

    def __init__(self, sentences, model_type, *cat_lists):
        self.sentences = sentences
        self.model_type = model_type
        self.num_cats = len(cat_lists)
        kwargs = dict(zip(["class" + str(i + 1) + "_target" for i in range(self.num_cats)], cat_lists))
        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        output = dict()
        for i in self.__dict__:
            if i in ["num_cats", "model_type"]:
                pass
            elif i == "sentences":
                output["sample"] = self.sentences[item]
            else:
                if "reg" in self.model_type:
                    output[i] = torch.tensor(self.__dict__[i][item], dtype=torch.float32)
                elif "class" in self.model_type:
                    output[i] = torch.tensor(self.__dict__[i][item], dtype=torch.long)
        return output


def training(data_loader, model, optimizer, device, scheduler, model_type, average="macro"):
    """ training
    This function takes care of the NN training for a model with one or two losses.
    This code might be changed for recurrent models
        -  data_loader: pytorch.DataLoader object
        -  model: BERT or another
        -  optimizer: optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        -  device: cuda
        -  scheduler: learning rate scheduler (torch.optim.lr_scheduler.StepLR()
    """
    # Initializing
    model.train()
    final_loss = 0

    if "class" in model_type:
        total_logits, total_target = [], []
    else:
        all_mse = []

    # loop over the data items
    pbar = tqdm(data_loader, total=len(data_loader))
    for data in pbar:
        pbar.set_description("Training the classifier ...")
        # Move value to device
        for key, value in data.items():
            if "class" in key:
                data[key] = value.to(device)
        # Initialize the gradients with zeros
        model.zero_grad()

        # Forward Propagation and Backward propagation
        logits, loss = model(**data)
        loss.backward()

        # Updates
        optimizer.step()
        scheduler.step()

        # accumulate the loss for the BP
        final_loss += loss.item()

        # Convert into our desired form
        logits = np.argmax(logits.detach().cpu().numpy(), axis=1)
        class1_target = data["class1_target"].detach().cpu().numpy()

        if "class" in model_type:
            total_logits.extend(logits)
            total_target.extend(class1_target)
        if "reg" in model_type:
            all_mse.append(mean_squared_error(class1_target, logits))

    if "class" in model_type:
        # numpy version
        total_target = np.asarray(total_target).reshape(-1)
        total_logits = np.asarray(total_logits).reshape(-1)
        # metrics
        confusion = confusion_matrix(total_target, total_logits)
        ACC = accuracy_score(total_target, total_logits)
        P = precision_score(total_target, total_logits, average=average, zero_division=0)
        R = recall_score(total_target, total_logits, average=average, zero_division=0)
        F1 = f1_score(total_target, total_logits, average=average, zero_division=0)
        return final_loss / len(data_loader), (ACC, P, R, F1), confusion
    if "reg" in model_type:
        mse = np.asarray([all_mse]).mean()
        return final_loss / len(data_loader), mse


def validation(data_loader, model, device, model_type, average="macro"):
    """ Validation phase
    This function takes care of the NN training for a model with one or two losses.
    This code might be changed for recurrent models
        -  data_loader: pytorch.DataLoader object
        -  model: BERT or another
        -  device: cuda
        -  model_type: str indicating if it is a classification problem or a regression one.
        -  scheduler: learning rate scheduler (torch.optim.lr_scheduler.StepLR()
    """
    model.eval()
    final_loss = 0

    if "class" in model_type:
        all_accuracies, all_f1, correct = [], [], 0
        total_logits, total_target = [], []
    else:
        all_mse = []


    # loop over the data items
    pbar = tqdm(data_loader, total=len(data_loader))
    for data in pbar:
        pbar.set_description("validation for the classifier ...")
        # Move value to device
        for key, value in data.items():
            if "class" in key:
                data[key] = value.to(device)
        # Forward Propagation
        with torch.no_grad():
            logits, loss = model(**data)

        # accumulate the loss for the BP
        final_loss += loss.item()

        # Convert into our desired form
        logits = np.argmax(logits.detach().cpu().numpy(), axis=1)
        class1_target = data["class1_target"].detach().cpu().numpy()

        if "class" in model_type:
            total_logits.extend(logits)
            total_target.extend(class1_target)
        if "reg" in model_type:
            all_mse.append(mean_squared_error(class1_target, logits))

    if "class" in model_type:
        # numpy version
        total_target = np.asarray(total_target).reshape(-1)
        total_logits = np.asarray(total_logits).reshape(-1)
        # metrics
        confusion = confusion_matrix(total_target, total_logits)
        ACC = accuracy_score(total_target, total_logits)
        P = precision_score(total_target, total_logits, average=average, zero_division=0)
        R = recall_score(total_target, total_logits, average=average, zero_division=0)
        F1 = f1_score(total_target, total_logits, average=average, zero_division=0)
        return final_loss / len(data_loader), (ACC, P, R, F1), confusion
    if "reg" in model_type:
        mse = np.asarray([all_mse]).mean()
        return final_loss / len(data_loader), mse
