import numpy as np
from tqdm import tqdm
import os, argparse
from data_loader import load_base_model

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--output_file", required=True, type=str, help='path of the rendered video sequence')
    parser.add_argument("--bs_path", required=True, type=str, help='path of the predictions')
    parser.add_argument("--base_models", required=True, type=str, help='path of base models')
    args = parser.parse_args()

    base_models = load_base_model(args.base_models)
    base_models = base_models[1:] - base_models[0] # - template
    base_models = base_models.reshape(55, -1)
    get_vertice = lambda x: x @ base_models

    bs = np.load(args.bs_path)
    bs = get_vertice(bs)
    np.save(args.output_file, bs)


if __name__=="__main__":
    main()
