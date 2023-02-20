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

import numpy as np
import os
import subprocess   # for ct file graph visualization

abs_file = __file__
work_dir = os.path.split(os.path.split(abs_file)[0])[0]
print("work_dir:{}".format(work_dir))


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
        save_embeddings_format="raw",
        save_results=True,
        save_frequency=2,   # save how many samples in a dict
        save_file_prefix="",
        threshold=0.5,
        allow_noncanonical_pairs=True,
        allow_visualization=False,
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
        #"""
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
                    if len(engine.state.output["pd"][key].shape) == 4:
                        pds = torch.softmax(engine.state.output["pd"][key].cpu().detach(), dim=1)
                    else:
                        pds = torch.sigmoid(engine.state.output["pd"][key].cpu().detach())
                    #pds = torch.softmax(engine.state.output["pd"][key].cpu().detach(), dim=1)
                batch_size = pds.shape[0]

                for i in range(batch_size):
                    id = engine.state.output["name"][i]
                    seq = engine.state.output["string"][i]
                    length = engine.state.output["length"][i]
                    if "contact" in key:
                        pd_numpy = pds[i][0:length, 0:length].numpy()  # pd_numpy = pds[i][1, 0:length, 0:length].numpy()
                    elif "r-ss" in key:
                        pd_numpy = pds[i][0:length, 0:length].numpy()
                        # save post-processed map
                        post_full_numpy, post_without_mlets_numpy, multiplet_list = postprocess(pd_numpy, seq, threshold=threshold, allow_nc=allow_noncanonical_pairs)
                        save_post_dir = os.path.join(cfg.SOLVER.OUTPUT_DIR, key + "_post_npy_full_bps")
                        if not os.path.exists(save_post_dir):
                            os.makedirs(save_post_dir)
                        np.save(os.path.join(save_post_dir, "{}".format(id)), post_full_numpy)
                        save_post_dir = os.path.join(cfg.SOLVER.OUTPUT_DIR, key + "_post_npy_no_mlets")
                        if not os.path.exists(save_post_dir):
                            os.makedirs(save_post_dir)
                        np.save(os.path.join(save_post_dir, "{}".format(id)), post_without_mlets_numpy)

                        # save ct file, with post_without_mbp_numpy
                        save_predCT_dir = os.path.join(cfg.SOLVER.OUTPUT_DIR, "pred_ct")
                        if not os.path.exists(save_predCT_dir):
                            os.makedirs(save_predCT_dir)
                        matrix2ct(post_without_mlets_numpy, seq, id, save_predCT_dir, threshold=threshold, with_post=False, nc=allow_noncanonical_pairs)


                    elif key == "representations":
                        pd_numpy = pds[i][1:1 + engine.state.output["length"][i]]
                        print(pd_numpy.shape)
                        if pd_numpy.shape[0] != engine.state.output["length"][i]:
                            raise Exception
                        # CJY
                        if save_embeddings_format == "mean":
                            pd_numpy = pd_numpy.mean(dim=0)   # mean  along sequence direction
                        pd_numpy = pd_numpy.numpy()

                    if save_frequency == 1:
                        #print(save_pd_dir)
                        np.save(os.path.join(save_pd_dir, "{}".format(id.replace(" ", "_"))), pd_numpy)
                    else:
                        engine.state.result_cache[key][id] = pd_numpy
                        engine.state.result_count[key] += 1
                        #print(engine.state.result_count[key])

                        if engine.state.result_count[key] % save_frequency == 0:
                            np.save(os.path.join(
                                save_pd_dir, "{}-{}-collection-{}-{}".format(save_file_prefix, key, save_frequency, engine.state.result_count[key] // save_frequency)).strip("-"),
                                engine.state.result_cache[key]
                            )
                            engine.state.result_cache[key] = {}

                if save_frequency != 1 and engine.state.iteration == len(test_loader) and engine.state.result_cache[key] != {}:
                    np.save(
                        os.path.join(save_pd_dir, "{}-{}-collection-{}-{}-left{}".format(save_file_prefix, key, save_frequency, engine.state.result_count[key] // save_frequency, engine.state.result_count[key])).strip("-"),
                        engine.state.result_cache[key]
                    )
                    engine.state.result_cache[key] = {}
        #"""

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_inference_results(engine):
        info = "Finish Prediction\n"
        logger.info(info.replace("'", "").strip("\n"))

    evaluator.run(test_loader)

    return Eval_Record


def matrix2ct(prob_map, seq, seq_id, ct_dir, threshold=0.5, with_post=False, nc=False):
    """
    :param contact: binary matrix numpy
    :param seq: string
    :return:
    """
    # 1.process matrix to make it obey the required constraints (maybe need sequence string)
    if with_post == True:
        contact = postprocess(prob_map, threshold=threshold, seq=seq, nc=nc)
    else:
        if threshold > 0:
            contact = (prob_map > threshold)
        else:
            contact = prob_map

    # 2.write ct file
    seq_len = len(seq)
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

    if os.path.exists(ct_dir) != True:
        os.makedirs(ct_dir)
    ct_file = os.path.join(ct_dir, seq_id+".ct")

    with open(ct_file, "w") as f:
        f.write("{}\t{}\n".format(seq_len, seq_id))  # header
        for i in range(seq_len):
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(first_col[i], second_col[i], third_col[i], fourth_col[i], fifth_col[i], last_col[i]))



def postprocess(prob_map, seq, threshold=0.5, allow_nc=True):
    # we suppose that probmay cantains values range from [0,1], so is the threshold
    canonical_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']

    # candidate 1: threshold  (we obatin the full contact matrix)
    prob_map = prob_map * (1 - np.eye(prob_map.shape[0]))  # no  care about the diagonal
    pred_map = (prob_map > threshold)

    # candidate 2: split the multiplets, resulting in cm without multiplets. Also filter the non-canonical pairs
    # when several pairs are conflict by presenting in the same row or column, we choose the one with highest score.
    seq_len = len(seq)
    x_array, y_array = np.nonzero(pred_map)
    prob_array = []
    for i in range(x_array.shape[0]):
        prob_array.append(prob_map[x_array[i], y_array[i]])
    prob_array = np.array(prob_array)

    sort_index = np.argsort(-prob_array)

    mask_map = np.zeros_like(pred_map)
    already_x = set()
    already_y = set()
    multiplet_list = []
    for index in sort_index:
        x = x_array[index]
        y = y_array[index]

        # # no sharp stem-loop
        if abs(x - y) <= 1:    # when <=1, allow 1 element loop
            continue

        seq_pair = seq[x] + seq[y]
        if seq_pair not in canonical_pairs and allow_nc == False:
            # print(seq_pair)
            continue
            pass

        if x in already_x or y in already_y:  # this is conflict
            multiplet_list.append([x+1,y+1])
            continue
        else:
            mask_map[x, y] = 1
            already_x.add(x)
            already_y.add(y)

    pred_map_without_multiplets = pred_map * mask_map

    return pred_map, pred_map_without_multiplets, multiplet_list
