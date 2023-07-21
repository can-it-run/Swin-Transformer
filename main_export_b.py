# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import numpy as np
import pdb
import inspect
import torch
import os
import json
import argparse
from config import get_config
from models import build_model
from logger import create_logger
from utils.gen_shell import generate


def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used (deprecated!)",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument(
        "--fused_window_process",
        action="store_true",
        help="Fused window shift & window partition, similar for reversed part.",
    )
    parser.add_argument(
        "--fused_layernorm", action="store_true", help="Use fused layernorm."
    )
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument(
        "--optim",
        type=str,
        help="overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.",
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    sd = torch.load(
        "/workspace/model_swin/swin_base_patch4_window7_224.pth", map_location="cpu"
    )

    model.load_state_dict(sd['model'])
    # 创建一个形状为 (1, 4, 4, 4) 的输入张量
    input_tensor = torch.rand(1, 3, 224, 224)
    # input_tensor = torch.tensor((res['Input'])).reshape(4,16,7,6,10)
    # traced = torch.jit.trace(model, (input_tensor,))
    # data = {"x.1": input_tensor.numpy()}
    # print(input_tensor.shape)
    # np.savez("../workspace_swins/data.npz", **data)
    # os.makedirs(f"./cali_data", exist_ok=True)
    # np.savez(f"./cali_data/data.npz", **data)

    # print(traced.graph)
    # torch.jit.save(traced, "../workspace_swins/swin-model.pt")
    print(inspect.getsourcefile(type(model)))
    print("downsample:", type(model.layers[0].downsample))
    print(inspect.getsourcefile(type(model.layers[0].downsample)))
    print("over")
    # pdb.set_trace()
    # exit()
    generate("swin_b", model, [input_tensor], "/workspace/model_swin/swin_b_workspace")


if __name__ == "__main__":
    args, config = parse_option()

    config.defrost()    
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}"
    )

    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
