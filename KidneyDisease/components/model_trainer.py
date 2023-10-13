import os
import torch, torchvision
from pathlib import Path
import random
from PIL import Image
from torch import nn
from torchvision import transforms

from KidneyDisease.utils.helper_functions import *
from KidneyDisease.config.configuration import ModelTrainerConfig

# Setting up the device agnostic code.
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config
        
    def trainer(self):
        # Location of the data
        loc = self.config.data
        
        # Transforming the data and turning into tensor format
        data_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
        ])
        
        # Custom data loader for pytorch
        data = ImageFolderCustom(targ_dir=loc,
                                 transform=data_transform)
        
        ## Splitting the data into train and test data for training.
        train_size = int(0.70 * len(data))
        test_size = len(data) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

        # Setup batch size and number of workers
        BATCH_SIZE = 64
        print(f"Creating DataLoader's with batch size {BATCH_SIZE}")

        # Turn train and test custom Dataset's into DataLoader's
        from torch.utils.data import DataLoader
        train_dataloader_custom = DataLoader(dataset=train_dataset, # use custom created train Dataset
                                            batch_size=BATCH_SIZE, # how many samples per batch?
                                            num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                            shuffle=True) # shuffle the data?

        test_dataloader_custom = DataLoader(dataset=test_dataset, # use custom created test Dataset
                                            batch_size=BATCH_SIZE,
                                            num_workers=0,
                                            shuffle=False) # don't usually need to shuffle testing data    
                
        # 1. Setup pretrained EffNetB2 weights
        effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

        # 2. Get EffNetB2 transforms
        effnetb2_transforms = effnetb2_weights.transforms()

        # 3. Setup pretrained model
        effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights) # could also use weights="DEFAULT"

        # 4. Freeze the base layers in the model (this will freeze all layers to begin with)
        for param in effnetb2.parameters():
            param.requires_grad = False
            
        # 5. Update the classifier head
        effnetb2.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # keep dropout layer same
            nn.Linear(in_features=1408, # keep in_features same
                    out_features=4)) # change out_features to suit our number of classes   
                    
        # Define loss and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=effnetb2.parameters(),
                                    lr=1e-3) 

        # Start the timer
        from timeit import default_timer as timer
        start_time = timer()

        # Setup training and save the results
        results = train(model=effnetb2.to(device),
                            train_dataloader=train_dataloader_custom,
                            test_dataloader=test_dataloader_custom,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=1)

        # End the timer and print out how long it took
        end_time = timer()
        print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds") 
        
        torch.save(effnetb2, 'artifacts/models/model.pth')
    