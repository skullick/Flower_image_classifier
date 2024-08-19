import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def parseArgs():
    parser = argparse.ArgumentParser(description='Training neural network')
    parser.add_argument('--data_dir', help='Path to dataset')
    parser.add_argument('--save_dir', help='Path to save directory', default='checkpoint.pth')
    parser.add_argument('--arch', help='Network architecture', default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--epochs', help='Number of epochs', default='5')
    parser.add_argument('--learning_rate', help='Learning rate', default='0.001')
    parser.add_argument('--hidden_units', help='Number of hidden units', default='512')
    parser.add_argument('--gpu', help='Use GPU', action='store_true')
    return parser.parse_args()

def loadData(path):
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validateloader = torch.utils.data.DataLoader(validate_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_data, trainloader, validateloader, testloader

def build(arch, hidden_units):
    hidden_units = int(hidden_units)
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
              nn.Linear(25088, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
    elif arch == "densenet121":
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential( nn.Linear(1024, hidden_units),
                                  nn.Dropout(p=0.6),
                                  nn.ReLU(),
                                  nn.Linear(hidden_units, 102),
                                  nn.LogSoftmax(dim=1))
        
    
    print("======== FINISHED BUILDING ========")
    
    return model
    
    
    
def train(model, trainloader, validateloader, epochs, learning_rate, gpu):
    print("1")
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("2")
    else:
        device = torch.device("cpu")
        print("3")
    learning_rate = float(learning_rate)
    epochs = int(epochs)
    criterion = nn.NLLLoss(); print("4")
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate); print("5")
    model.to(device)
    print("Start training...")
    
        
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % 5 == 0:
                validate_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validateloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validate_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"========= Epoch {epoch+1}/{epochs} ========="
                      f"\nTrain loss:        {running_loss/5:.3f}.. "
                      f"\nValidate loss:     {validate_loss/len(validateloader):.3f}.. "
                      f"\nValidate accuracy: {accuracy/len(validateloader):.3f}")
                running_loss = 0
                model.train()
    print("======== FINISHED TRAINING ========")
    return model

def evaluate(model, testloader, gpu):
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    steps = 0
    for epoch in range(epochs):
        if steps % 5 == 0:
            accuracy = 0
            test_loss = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"========= Epoch {epoch+1}/{epochs} ========="
                  f"\nTest loss:     {test_loss/len(testloader):.3f}.. "
                  f"\nTest accuracy: {accuracy/len(testloader):.3f}")
    print("======== FINISHED EVALUATING ========")
    
def save(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    checkpoint = {
    'arch': architecture,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'hidden_units': hidden_units,
    'model_state_dict': model.state_dict(),
    'map_class_idx': model.class_to_idx
}

    torch.save(checkpoint, save_dir)
    print(f"...Saved to {save_dir}...")
    
    


def main():
    args = parseArgs()
    gpu = False if args.gpu is None else True 
    train_data, trainloader, validateloader, testloader = loadData(args.data_dir)
    model = build(args.arch, args.hidden_units)
    model.class_to_idx = train_data.class_to_idx
    model = train(model, trainloader, validateloader, args.epochs, args.learning_rate, args.gpu)
    evaluate(model, testloader, args.gpu)
    save(model, args.arch, args.hidden_units, args.epochs, args.learning_rate, args.save_dir)
    
if __name__ == '__main__':
    main()
    
 #python train.py --data_dir flowers --save_dir checkpoint.pth --arch vgg16 --epochs 5 --learning_rate 0.001 --gpu