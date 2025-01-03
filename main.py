import argparse
from utils import *
from torchvision.models import vgg19, VGG19_Weights
from attack import *
from torch import optim as optim
import torch
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
from colorl2 import *
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
parser = argparse.ArgumentParser()
parser.add_argument("--content_image_folder", dest='content_image_folder',
                    help="Path to the content image",default='./data/sub_imagenet/img')
parser.add_argument("--style_image_path",   dest='style_image_path',
                    help="Path to the style image",default='./data/image.png')

parser.add_argument("--content_weight",     dest='content_weight',      nargs='?', type=float,
                    help="weight of content loss", default=10)
parser.add_argument("--style_weight",       dest='style_weight',        nargs='?', type=float,
                    help="weight of style loss", default=1e6)
parser.add_argument("--attack_weight",      dest='attack_weight',       nargs='?', type=float,
                    help="weight of attack loss", default=1e2)
parser.add_argument("--target_label",       dest='target_label',        nargs='?', type = int,
                    help="The target label for target attack", default=498)
parser.add_argument("--true_label",       dest='true_label',        nargs='?', type = int,
                    help="The target label for target attack", default=8)
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int,
                    help='maximum image iteration', default=300)

args = parser.parse_args()

vgg = vgg19(weights=VGG19_Weights.DEFAULT)  
vgg.eval() 

cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=args.max_iter,
                       style_weight=args.style_weight, content_weight=args.content_weight,
                       attack_weight=args.attack_weight, target_image=None, name=None):
    style_losses_list = []
    content_losses_list = []
    attack_losses_list = []
    hvs_losses_list = []
    total_losses_list = []
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])
    _, args.true_label = predict(vgg, input_img)
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            attack_score = 0
            pred, _ = predict(vgg, input_img)
            attack_score = targeted_attack_loss(pred, args.true_label, args.target_label, attack_weight)
            hvs_score = ciede2000_loss(target_image, input_img)
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_losses_list.append(style_score.item())
            content_losses_list.append(content_score.item())
            attack_losses_list.append(attack_score.item())
            hvs_losses_list.append(hvs_score.item())
            style_score *= style_weight
            content_score *= content_weight
            hvs_score *= 500
            loss = style_score + content_score + attack_score + hvs_score
            loss.backward()
            run[0] += 1

            if run[0] % 50 == 0:
                print("Epoch {}:".format(run[0]))
                print('Style Loss : {:4f} Content Loss: {:4f} HVS Loss: {:4f} Attack Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), hvs_score.item(), attack_score.item()))
                pred, _ = print_prob(pred.detach().cpu().numpy(), './synset.txt')
                print('Current prediction: {}'.format(pred))
                print()
            return style_score + content_score + attack_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    # plt.figure(figsize=(10, 7))
    # plt.plot(style_losses_list, label='Style Loss')
    # plt.plot(content_losses_list, label='Content Loss')
    # plt.plot(attack_losses_list, label='Attack Loss')
    # plt.plot(hvs_losses_list, label='HVS Loss')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.title('Loss Curves')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'data/adversarial_result/{name}.png')
    return input_img

if __name__ == "__main__":
    style_img = image_loader(args.style_image_path)
    for image in os.listdir(args.content_image_folder):
        image = os.path.join(args.content_image_folder, image).replace("/", "\\")
        print(image)
        prob, _ = predict(vgg, image_loader(image).to(device))
        pred, prob = print_prob(prob.detach().cpu().numpy(), './synset.txt')
        label = ''.join(pred.split(" ")[1:])
        prob = round(float(prob), 2)
        add_text_to_image(image, "./data/result/" + image.split("\\")[-1], label.split(",")[0] + "," + str(prob))
        print(pred, prob)
        content_img = image_loader(image)
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        input_img = content_img.clone()
        target_image = load_image('./data/adversarial_result/' + image.split("\\")[-1]).to(device)
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, target_image=target_image, name=image.split("\\")[-1].split(".")[0])

        prob, _ = predict(vgg, output.to(device))
        output = output.permute(0, 2, 3, 1).detach().cpu().numpy()[0] * 255 
        output = output.astype(np.uint8) 
        output = Image.fromarray(output)
        output.save('./data/adversarial_result/' + "monic_" + image.split("\\")[-1])
        # output.save('./data/adversarial_result/' + image.split("\\")[-1])
        pred, prob = print_prob(prob.detach().cpu().numpy(), './synset.txt')
        label = ''.join(pred.split(" ")[1:])
        prob = round(float(prob), 2)
        print(label, prob)
        # add_text_to_image(output, './data/adversarial_result/text/' + image.split("\\")[-1], label.split(",")[0] + "," + str(prob), font_size=20)
        ssim, psnr = calculate_ssim_psnr('./data/adversarial_result/' + "monic_" + image.split("\\")[-1], './data/adversarial_result/' + image.split("\\")[-1])
        print(ssim, psnr)
        add_text_to_image(output, './data/adversarial_result/text/' + image.split("\\")[-1], label.split(",")[0] + "," + str(prob) + '\npsnr = ' + str(round(psnr,4)) + '\nssim = ' + str(round(ssim,4)), font_size=40)
        add_text_to_image(output, './data/adversarial_result/text/' + image.split("\\")[-1], '')
        
        # plt.figure()
        # plt.imshow(output, title='Output Image')
        # plt.ioff()
        # plt.savefig("./data/adversarial_result")
    
