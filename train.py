import torch
from config import *
from transforms import *
from model import *
from data import *
import mlflow
import time
from tqdm.auto import tqdm
import argparse
import sys


def train(model, trainloader, optimizer, criterion,scheduler):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0.0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = torch.stack([T(img) for img in image])
        image = normalize_pretrained(image)
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        scheduler.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            # image = torch.stack([T(img) for img in image])
            image = normalize_pretrained(image)
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

def main(args):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data", required=True,help="path to root directory")
    ap.add_argument("-o", "--output", required=True,help="path to output image")
    ap.add_argument("-sz", "--img_sz", required=True,help="image size")
    ap.add_argument("-b", "--batch", required=True,help="batch size")
    ap.add_argument("-e", "--epochs", required=True,help="epochs")
    ap.add_argument("-mn", "--model_name", required=True,help="model name")
    ap.add_argument("-u", "--mlflow_url", required=True,help="mlflow url")
    ap.add_argument("-exp", "--exp_name", required=True,help="experiment name")
    ap.add_argument("-lr", "--learning_rate", required=True,help="learning rate")
    args = ap.parse_args(args)
    return [args.data+"/train",args.data+"/val",args.output,args.img_sz,args.batch,args.epochs,args.model_name,args.mlflow_url,args.exp_name,args.learning_rate]




if __name__ == '__main__':

    train_dir,val_dir,dest,IMAGE_SIZE,BATCH_SIZE,EPOCHS,model_name,MLFLOW_URL,EXP_NAME,lr= main(sys.argv[1:])


    dataset_train, dataset_valid, dataset_classes = get_datasets(train_dir,val_dir,IMAGE_SIZE,True)

    train_loader, valid_loader = get_data_loaders(BATCH_SIZE,2,dataset_train, dataset_valid)

    
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = build_model(
        pretrained=True, 
        fine_tune=True, 
        num_classes=len(dataset_classes)
    ).to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    
    criterion = torch.nn.CrossEntropyLoss()
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    best_accuracy=0.0
    best_model=model

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(EXP_NAME)
    experiment = mlflow.get_experiment_by_name(EXP_NAME)



    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=EXP_NAME):

        for epoch in range(EPOCHS):
            epoch_time = time.time()
            # print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion,scheduler)
            #print(exp_lr_scheduler.get_last_lr())
            valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,criterion)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            # print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            # print('-'*50)

            if best_accuracy < valid_epoch_acc:
                best_model=model
                best_accuracy=valid_epoch_acc
                mlflow.pytorch.log_model(model, "best")  

            # print('estimated time of completion:',(time.time()-epoch_time)*(epochs-epoch-1)/3600,' hrs ')

            mlflow.log_metric("training loss", f"{train_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("training accuracy", f"{train_epoch_acc:3f}", step=epoch)

            mlflow.log_metric("validation loss", f"{valid_epoch_loss:3f}", step=epoch)
            mlflow.log_metric("validation accuracy", f"{valid_epoch_acc:3f}", step=epoch)
            mlflow.pytorch.log_model(model, f"{epoch}")  

    print('TRAINING COMPLETE')
