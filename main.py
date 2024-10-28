import argparse
from utils import *
from torchvision.models import vgg19, VGG19_Weights
from attack import *
from torch import optim as optim
from matplotlib import pyplot as plt
import os
import random

random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--content_image_folder", dest='content_image_folder',
                    help="Path to the content image",default='./data/images/content')
parser.add_argument("--style_image_path",   dest='style_image_path',
                    help="Path to the style image",default='./data/images/style/picasso.jpg')

parser.add_argument("--content_weight",     dest='content_weight',      nargs='?', type=float,
                    help="weight of content loss", default=5)
parser.add_argument("--style_weight",       dest='style_weight',        nargs='?', type=float,
                    help="weight of style loss", default=1e4)
parser.add_argument("--attack_weight",      dest='attack_weight',       nargs='?', type=float,
                    help="weight of attack loss", default=1e4)
parser.add_argument("--target_label",       dest='target_label',        nargs='?', type = int,
                    help="The target label for target attack", default=498)
parser.add_argument("--true_label",       dest='true_label',        nargs='?', type = int,
                    help="The target label for target attack", default=8)
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int,
                    help='maximum image iteration', default=5000)

args = parser.parse_args()

vgg = vgg19(weights=VGG19_Weights.DEFAULT)  
vgg.eval() 

cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=args.max_iter,
                       style_weight=args.style_weight, content_weight=args.content_weight,
                       attack_weight=args.attack_weight):

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = optimizer = optim.Adam([input_img], lr=0.01)
    _, args.true_label = predict(vgg, input_img)
    # print(input_img)
    for i in range(num_steps + 1):

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            attack_score = 0
            pred, _ = predict(vgg, input_img)
            attack_score += targeted_attack_loss(pred, args.true_label, args.target_label, attack_weight)
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score + attack_score
            # loss = style_score + content_score
            loss.backward()
            if i % 50 == 0:
                print("Epoch {}:".format(i))
                print('Style Loss : {:4f} Content Loss: {:4f} Attack Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), attack_score.item()))
                pred, _ = print_prob(pred.detach().cpu().numpy(), './synset.txt')
                print('Current prediction: {}'.format(pred))
                print()
            # return style_score + content_score
            return style_score + content_score + attack_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

if __name__ == "__main__":
    style_img = image_loader(args.style_image_path)
    for image in os.listdir(args.content_image_folder):

        image = os.path.join(args.content_image_folder, image).replace("/", "\\")
        print(image)
        prob, _ = predict(vgg, image_loader(image).to(device))
        # print(prob.detach().cpu().numpy())
        pred, prob = print_prob(prob.detach().cpu().numpy(), './synset.txt')
        label = ''.join(pred.split(" ")[1:])
        prob = round(float(prob), 2)
        add_text_to_image(image, "./data/result/" + image.split("\\")[-1], label.split(",")[0] + "," + str(prob))
        print(pred, prob)
        content_img = image_loader(image)
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        input_img = content_img.clone()
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

        prob, _ = predict(vgg, output.to(device))
        # print(output)
        output = output.permute(0, 2, 3, 1).detach().cpu().numpy()[0] * 255 
        output = output.astype(np.uint8) 
        output = Image.fromarray(output)
        output.save('./data/adversarial_result/' + image.split("\\")[-1])
        pred, prob = print_prob(prob.detach().cpu().numpy(), './synset.txt')
        label = ''.join(pred.split(" ")[1:])
        prob = round(float(prob), 2)
        print(label, prob)
        add_text_to_image(output, './data/adversarial_result/text/' + image.split("\\")[-1], label.split(",")[0] + "," + str(prob), font_size=20)
        # add_text_to_image(output, './data/adversarial_result/text/' + image.split("\\")[-1], '')
        
        # plt.figure()
        # plt.imshow(output, title='Output Image')
        # plt.ioff()
        # plt.savefig("./data/adversarial_result")
    
