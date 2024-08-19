import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import json


def parseArgs():
    parser = argparse.ArgumentParser(description='Flower Image Classifier')
    parser.add_argument('path', help='Image path')
    parser.add_argument('checkpoint', help='Network checkpoint')
    parser.add_argument('--top_k', help='Return top k most likely classes', default='3')
    parser.add_argument('--category_names', help='Map categories to real names', default='cat_to_name.json')
    parser.add_argument('--gpu', help='Use GPU ', action='store_true')
    return parser.parse_args()

def load(filepath):
    checkpoint = torch.load(filepath)
    model = build(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['map_class_idx']
    
    return model

def process_image(image_path):
    img = Image.open(image_path)
    img_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    return img_transform(img)
    

def predict(image_path, model, topk):
    model.eval()
    with torch.no_grad():
        logps = model.forward(process_image(image_path).unsqueeze(0)).cpu()
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        idx_to_class = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
        
        idx_to_class = {value : key for key, value in model.class_to_idx.items()} 
        classes = [idx_to_class[i] for i in labels.numpy()[0]]
        
        
        return probs.numpy()[0], classes


def main():
    model = load(args.path)
    print(model)
    probs, predict_classes = predict(data_management.process_image(args.image_path), model, top_k)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[str(index)] for index in classes]
    
    print(f"{labels[0]} - Probability: {probs[0]:.3f}\n")"
    print("================= Top K =================")
    for label in labels:
        print(f"{labels[0]} - Probability: {probs[0]:.3f}")
    print("================= Top K =================")
    
if __name__ == "__main__":
    main()