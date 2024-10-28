import torch
from torchvision.transforms import transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

preprocess = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224),  
    transforms.ToTensor()])

def image_loader(image_name):
    if isinstance(image_name, str):
        image = Image.open(image_name)
    else:
        image = image_name
    image = preprocess(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(activations):
    batch_size, height, width, num_channels = activations.size()
    gram_matrix = activations.permute(0, 3, 1, 2)
    gram_matrix = gram_matrix.reshape(num_channels * batch_size, width * height)
    gram_matrix = torch.mm(gram_matrix, gram_matrix.t())
    return gram_matrix.div(batch_size * height * width * num_channels)

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    # print(("Top5: ", top5))
    return top1, prob[pred[0]]

compose = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224)])
 
def add_text_to_image(image_path, output_path, text, font_path='arial.ttf', font_size=30, text_color=(255, 255, 255)):   
    if isinstance(image_path, str):
        image = Image.open(image_path)  
    else:
        image = image_path
    image = compose(image)
    draw = ImageDraw.Draw(image)  
    font = ImageFont.truetype(font_path, font_size)  
    text_width = draw.textlength(text, font)  
    image_width, image_height = image.size  
    x = (image_width - text_width) / 2
    y = image_height  
    vertical_spacing = 40
    y -= vertical_spacing  
    draw.text((x, y), text, font=font, fill=text_color)   
    image.save(output_path)  
