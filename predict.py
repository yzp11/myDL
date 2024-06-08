
import os

import torch
from PIL import Image

from datasets.Flowers import data_transform
from tools.utils import create_model, model_parallel
from configs.load_configs import load_configs


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    image_path = "./data/validation/daisy/11642632_1e7627a2cc.jpg"
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)

    image = Image.open(image_path)

    image = data_transform["val"](image)
    image = torch.unsqueeze(image, dim=0)


    model = create_model(args=args)
    model = model_parallel(args, model)
    model.to(device)

    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    weights_dict = torch.load(args.weights, map_location=device)
    print(model.load_state_dict(weights_dict))


    model.eval()
    with torch.no_grad():
        
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        index = torch.argmax(predict).numpy()

    print("prediction: {}   prob: {:.3}\n".format(args.label_name[index],
                                                predict[index].numpy()))
    for i in range(len(predict)):
        print("class: {}   prob: {:.3}".format(args.label_name[i],
                                               predict[i].numpy()))


if __name__ == '__main__':
    main(load_configs())