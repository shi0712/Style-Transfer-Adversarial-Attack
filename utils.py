import torch
from torchvision.transforms import transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def calculate_ssim_psnr(image_path1, image_path2):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    if img1.mode == 'RGB':
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        channels = 3
    elif img1.mode == 'L':
        channels = 1
    else:
        raise ValueError(f"不支持的图像模式: {img1.mode}")
    if img1.size != img2.size:
        raise ValueError("两张图片的尺寸不一致！")
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    if channels == 3:
        ssim_total = 0
        psnr_total = 0
        for i in range(3):
            ssim, _ = compare_ssim(img1_np[:, :, i], img2_np[:, :, i], full=True)
            psnr = compare_psnr(img1_np[:, :, i], img2_np[:, :, i])
            ssim_total += ssim
            psnr_total += psnr
        ssim_average = ssim_total / 3
        psnr_average = psnr_total / 3
    else:
        ssim_average, _ = compare_ssim(img1_np, img2_np, full=True)
        psnr_average = compare_psnr(img1_np, img2_np)
    
    return ssim_average, psnr_average


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

preprocess = transforms.Compose([
    transforms.Resize([512,512]),  
    transforms.ToTensor()])

def image_loader(image_name):
    if isinstance(image_name, str):
        image = Image.open(image_name)
    else:
        image = image_name
    image = preprocess(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d) 
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    # print(("Top5: ", top5))
    return top1, prob[pred[0]]

compose = transforms.Compose([
    transforms.Resize([512,512])])
 
def add_text_to_image(image_path, output_path, text, font_path='arial.ttf', font_size=30, text_color=(255, 255, 255)):   
    if isinstance(image_path, str):
        image = Image.open(image_path)  
    else:
        image = image_path
    image = compose(image)
    draw = ImageDraw.Draw(image)  
    font = ImageFont.truetype(font_path, font_size)  
    text_width = draw.textlength(text.split("\n")[0], font)  
    image_width, image_height = image.size  
    x = (image_width - text_width) / 2
    y = image_height  
    vertical_spacing = 160
    y -= vertical_spacing  
    draw.text((x, y), text, font=font, fill=text_color)   
    image.save(output_path)  
