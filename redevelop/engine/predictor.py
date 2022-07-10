# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.metrics import Average

import os

def print_format(dic):
    formative_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            formative_dict[key] = print_format(dic[key])
        else:
            formative_dict[key] = "{:.3f}".format(dic[key])
    return formative_dict


def create_supervised_predictor(model, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device != "cpu":
        if next(model.parameters()).is_cuda:
            pass
        else:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # fetch data
            data, anns = batch

            # place data and ann in CUDA
            if torch.cuda.device_count() >= 1:
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(device)
                    elif isinstance(data[key], list) and isinstance(data[key][0], torch.Tensor):
                        data[key] = [d.to(device) for d in data[key]]

                for key in anns.keys():
                    if isinstance(anns[key], torch.Tensor):
                        anns[key] = anns[key].to(device)
                    elif isinstance(anns[key], list) and isinstance(anns[key][0], torch.Tensor):
                        anns[key] = [d.to(device) for d in anns[key]]

            # forward propagation
            results = model(data)

            return {"pd": results, "name": data["description"], "string": data["string"],
                    "length": data["length"], "depth": data["depth"]}

    engine = Engine(_inference)

    return engine

def do_prediction(
        cfg,
        model,
        test_loader,
        save_embeddings=False,
        save_results=True,
        save_frequency=2,   # save how many samples in a dict
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("prediction.inference")
    logging._warn_preinit_stderr = 0
    logger.info("Enter inferencing for Custom set")

    # 1.Create engine
    evaluator = create_supervised_predictor(model, device=device)

    Eval_Record = {}

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if engine.state.iteration % 1 == 0:
            print("Iteration[{}/{}]".format(engine.state.iteration, len(test_loader)))
        pass

    if save_results == True:
        import numpy as np
        evaluator.state.result_cache = {}
        evaluator.state.result_count = {}
        @evaluator.on(Events.ITERATION_COMPLETED)
        def save_results(engine):
            for key in engine.state.output["pd"].keys():
                if key == "logits" or key == "rl":
                    continue
                elif key == "representations" and save_embeddings != True:
                    continue
                else:
                    if save_frequency > 1 and key not in engine.state.result_cache:
                        engine.state.result_cache[key] = {}
                        engine.state.result_count[key] = 0

                save_pd_dir = os.path.join(cfg.SOLVER.OUTPUT_DIR, key)
                if os.path.exists(save_pd_dir) != True:
                    os.makedirs(save_pd_dir)

                if key == "representations":
                    pds = engine.state.output["pd"][key][12].cpu().detach()
                else:
                    pds = torch.softmax(engine.state.output["pd"][key].cpu().detach(), dim=1)
                batch_size = pds.shape[0]

                for i in range(batch_size):
                    id = engine.state.output["name"][i]
                    length = engine.state.output["length"][i]
                    if "contact" in key:
                        pd_numpy = pds[i][1, 0:length, 0:length].numpy()
                    elif "r-ss" in key:
                        pd_numpy = pds[i][1, 0:length, 0:length].numpy()
                        seq = engine.state.output["string"][i]
                        save_file = os.path.join(save_pd_dir, "{}.ct".format(id))
                        save_ss2ct(pd_numpy, seq, id, save_file, threshold=0.5)

                    elif key == "representations":
                        pd_numpy = pds[i][1:1 + engine.state.output["length"][i]]

                    if save_frequency == 1:
                        np.save(os.path.join(save_pd_dir, "{}".format(id)), pd_numpy)
                    else:
                        engine.state.result_cache[key][id] = pd_numpy
                        engine.state.result_count[key] += 1
                        #print(engine.state.result_count[key])
                        if engine.state.result_count[key] % save_frequency == 0:
                            np.save(os.path.join(save_pd_dir, "{}-collection-{}-{}".format(key, save_frequency, engine.state.result_count[key]//save_frequency)), engine.state.result_cache[key])
                            engine.state.result_cache[key] = {}


    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_inference_results(engine):
        info = "Finish Prediction\n"
        logger.info(info.replace("'", "").strip("\n"))

    evaluator.run(test_loader)

    return Eval_Record


import numpy as np
def preprocess_ss_map(prob_map, seq, threshold=0.5, nc=True):
    canonical_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']  # for farfar2

    # candidate 1: threshold
    contact = (prob_map > threshold)

    # notes: for e2efold, do not need this step
    prob_map = prob_map * (1 - np.eye(prob_map.shape[0]))

    seq_len = len(seq)

    x_array, y_array = np.nonzero(contact)
    prob_array = []
    for i in range(x_array.shape[0]):
        prob_array.append(prob_map[x_array[i], y_array[i]])
    prob_array = np.array(prob_array)

    sort_index = np.argsort(-prob_array)

    mask_map = np.zeros_like(contact)
    already_x = set()
    already_y = set()
    for index in sort_index:
        x = x_array[index]
        y = y_array[index]

        seq_pair = seq[x] + seq[y]
        if seq_pair not in canonical_pairs and nc == True:
            # print(seq_pair)
            continue
            pass

        if x in already_x or y in already_y:
            continue
        else:
            mask_map[x, y] = 1
            already_x.add(x)
            already_y.add(y)

    contact = contact * mask_map

    return contact



def save_ss2ct(prob_map, seq, seq_id, save_file, threshold=0.5):
    """
    :param contact: binary matrix numpy
    :param seq: string
    :return:
    generate ct file from ss npy
    """
    seq_len = len(seq)

    contact = preprocess_ss_map(prob_map, seq, threshold)
    #contact = preprocess_map_umap(prob_map, seq, threshold)  # umap

    structure = np.where(contact)
    pair_dict = dict()
    for i in range(seq_len):
        pair_dict[i] = -1
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]
    first_col = list(range(1, seq_len+1))
    second_col = list(seq)
    third_col = list(range(seq_len))
    fourth_col = list(range(2, seq_len+2))
    fifth_col = [pair_dict[i]+1 for i in range(seq_len)]
    last_col = list(range(1, seq_len+1))

    save_dir, _ = os.path.split(save_file)
    if os.path.exists(save_dir) != True:
        os.makedirs(save_dir)

    with open(save_file, "w") as f:
        f.write("{}\t{}\n".format(seq_len, seq_id))
        for i in range(seq_len):
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(first_col[i], second_col[i], third_col[i], fourth_col[i], fifth_col[i], last_col[i]))

    # save secondary structure
    #save_png_file = save_file.replace(".ct", ".png")
    #save_png(contact, save_png_file, vmin=-1, vmax=1)

    return contact