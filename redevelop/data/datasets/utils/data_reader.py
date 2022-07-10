from typing import Tuple, List
import string
import itertools
from Bio import SeqIO
import numpy as np
import os
import random
import pickle
import pandas as pd

class DataReader(object):
    def __init__(self, theme, data_type="seq", use_cache=0, cache_dir=None, pss_type=3):
        """
        :param theme: "protein", "rna", "dna"
        :param data_type: "seq", "msa", "cov", "seq+msa", "seq+cov"   # input is controled
        :param msa_nseq: Reads the first nseq sequences from an MSA file if it is in "msa" data type.
        Notes:
        1.cache
        actually we should use dict for the data.
        data ->
        """
        # This is an efficient way to delete lowercase characters and insertion characters from a string
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        self.translation = str.maketrans(deletekeys)
        self.theme = theme
        self.data_type = data_type.split("+")

        # cache for accelerating loading
        # It is reasonable for us to reserve data and anns together,
        # but it is hard to save msa because it will change for random subsampling each epoch.
        # Therefore, we only save anns here
        self.use_cache = use_cache   # 0: No Cache 1: Disk Cache 2: RAM Cache
        self.cache_dir = cache_dir
        if self.use_cache == 2:
            self.cache_dat = {}
            self.cache_ann = {}
        if self.use_cache > 0 and os.path.exists(self.cache_dir) != True:
            os.makedirs(self.cache_dir)

        # Task specific
        # 1. pss - protein secondary structure
        if pss_type == 3:
            self.rss_classes = ["H", "E", "C"]
            self.class_to_label = {"H": 0, "E": 1, "L": 2, "T": 2, "S": 2, "G": 2, "I": 2, "B": 2}
        elif pss_type == 8:
            self.rss_classes = ["H", "E", "L", "T", "S", "G", "I", "B"]
            self.class_to_label = {"H": 0, "E": 1, "L": 2, "T": 3, "S": 4, "G": 5, "I": 6, "B": 7}
        else:
            raise Exception("Wrong Type PSS class Type")


    def read_data(self, data_path, msa_nseq=1, msa_mode="top"):
        # load from cache (no need cache because of input will change under some case. place in msa reader)
        """
        _, filename = os.path.split(data_path)
        prename, ext = os.path.splitext(filename)
        cache_name = prename + ".dat"
        cache_path = os.path.join(self.cache_dir, cache_name)
        if self.use_cache == 2:
            if self.cache_dat.get(filename) is not None:
                data = self.cache_dat[filename]
                return data
        elif self.use_cache == 1:
            if os.path.exists(cache_path) == True:
                with open(cache_path, 'rb') as handle:
                    data = pickle.load(handle)
                return data
        elif self.use_cache == 0:
            pass
        else:
            raise Exception("Wrong Cache Type!")
        """

        # main body
        data = {}
        for key in self.data_type:
            if key == "seq":
                data["seq"] = self.read_seq(data_path["seq"])
            elif key == "msa":
                data["msa"] = self.read_msa(data_path["msa"], msa_nseq, msa_mode)
            elif key == "msa-cov":
                data["msa-cov"] = self.read_numpy(data_path["msa-cov"])
            elif key == "petfold-ss":
                data["petfold-ss"] = np.expand_dims(self.read_numpy(data_path["petfold-ss"]), axis=-1)
            else:
                raise Exception("Wrong Type!")

        # save into cache
        """
        if self.use_cache == 2:
            if self.cache_dat.get(filename) is None:
                self.cache_dat[filename] = data
        if self.use_cache >= 1:
            if os.path.exists(cache_path) != True:
                with open(cache_path, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """

        return data

    def read_anns(self, anns_path):
        # fetch filename prefix
        ann_path = list(anns_path.values())[0]
        _, filename = os.path.split(ann_path)
        prefix, ext = os.path.splitext(filename)

        # load from cache
        cache_name = prefix + ".ann"
        cache_path = os.path.join(self.cache_dir, cache_name)
        if self.use_cache == 2:  # from RAM
            if self.cache_ann.get(prefix) is not None:
                annotations = self.cache_ann[prefix]
                return annotations

        if self.use_cache >= 1:  # from disk (including the first epoch while use_cache=2)
            cache_path = os.path.join(self.cache_dir, cache_name)
            if os.path.exists(cache_path) == True:
                with open(cache_path, 'rb') as handle:
                    annotations = pickle.load(handle)

                if self.use_cache == 2:
                    if self.cache_ann.get(prefix) is None:
                        self.cache_ann[prefix] = annotations

                return annotations

        # main body
        annotations = {}
        if anns_path.get("r-ss") is not None:
            ann_path = anns_path["r-ss"]
            rss = self.read_rss(ann_path)
            annotations["r-ss"] = rss

        if anns_path.get("p-ss") is not None:
            ann_path = anns_path["p-ss"]
            pss = self.read_pss(ann_path)
            annotations["p-ss"] = pss

        if anns_path.get("p-2d") is not None:
            ann_path = anns_path["p-2d"]
            distance_map, omega_map, theta_map, phi_map = self.read_protein_2d_map(ann_path)
            annotations["p-dist"] = distance_map

            contact_map = distance_map * (distance_map < 0) + (distance_map < 8) * (distance_map >= 0)
            contact_map = contact_map.astype(np.int8)
            annotations["p-contact"] = contact_map

            # discretization
            dist_bin_map = self.discretize_dist(distance_map)
            annotations["p-dist-bin"] = dist_bin_map.astype(np.int8)
            no_contact_mask = dist_bin_map == 0
            omega_bin_map = self.discretize_omega(omega_map, no_contact_mask)
            annotations["p-omega-bin"] = omega_bin_map.astype(np.int8)
            theta_bin_map = self.discretize_theta(theta_map, no_contact_mask)
            annotations["p-theta-bin"] = theta_bin_map.astype(np.int8)
            phi_bin_map = self.discretize_phi(phi_map, no_contact_mask)
            annotations["p-phi-bin"] = phi_bin_map.astype(np.int8)

        if anns_path.get("r-dist") is not None:
            ann_path = anns_path["r-dist"]
            rna_distance = self.read_rna_distance(ann_path)
            annotations["r-dist"] = rna_distance

            #rna_contact = rna_distance * (rna_distance < 0) + (rna_distance < 8) * (rna_distance >= 0)
            rna_contact = -1 * (rna_distance < 0) + (rna_distance < 8) * (rna_distance >= 0)
            annotations["r-contact"] = rna_contact.astype(np.int8)

        # reserve filename
        annotations["filename"] = prefix

        # save into cache
        if self.use_cache == 2:
            if self.cache_ann.get(prefix) is None:
                self.cache_ann[prefix] = annotations

        if self.use_cache >= 1:
            if os.path.exists(cache_path) != True:
                with open(cache_path, 'wb') as handle:
                    pickle.dump(annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return annotations

    # data loading functions
    def read_seq(self, filename: str) -> Tuple[str, str]:
        """ Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))   # read the first line of MSA data

        if self.theme == "rna":
            formatted_seq = str(record.seq).upper().replace('T', "U").replace('~', "-")
        else:
            formatted_seq = str(record.seq)

        # sometimes the description is chaos
        prename, ext = os.path.splitext(os.path.split(filename)[1])

        name = prename
        #name = record.description

        return name, formatted_seq

    def remove_insertions(self, sequence: str) -> str:
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(self.translation)

    def read_msa(self, filename: str, nseq: int, mode="top") -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        if nseq == 0:
            # which means attempting to extract maximum nseq according to the length of seq and the memory of gpu.
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), 1):
                length = len(self.remove_insertions(str(record.seq))) + 1
            all_tokens = pow(2, 15)
            nseq = all_tokens // length

        _, name = os.path.split(filename)
        range_limit = nseq if mode == "top" else 100000
        # save or load full record (only save into RAM)
        if self.use_cache == 2:
            if self.cache_dat.get(name) is None:
                full_record = []
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), range_limit):
                    full_record.append((record.description, self.remove_insertions(str(record.seq))))
                self.cache_dat[name] = full_record
            else:
                full_record = self.cache_dat[name]
        else:
            full_record = []
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), range_limit):
                full_record.append((record.description, self.remove_insertions(str(record.seq))))
        full_depth = len(full_record)
        #print(full_depth)

        if mode == "top":
            sampled_record = full_record

        elif mode == "random":
            sample_depth = nseq
            if full_depth > sample_depth:
                index_list = range(1, full_depth)
                downsample_index_list = random.sample(index_list, k=sample_depth - 1)
                downsample_index_list.sort()
                downsample_index_list = [0] + downsample_index_list

                sampled_record = []
                for index in downsample_index_list:
                    sampled_record.append(full_record[index])
            else:
                sampled_record = full_record

        else:
            raise Exception("Without this type {} of sampling MSA")

        return sampled_record

    #def read_msa(self, filename: str, nseq: int) -> List[Tuple[str, str]]:
    #    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    #    return [(record.description, self.remove_insertions(str(record.seq)))
    #            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

    def read_numpy(self, npy_path):
        matrix = np.load(npy_path)

        return matrix


    # ann loading functions
    def read_protein_2d_map(self, ann2d_path):
        information_list = []
        with open(ann2d_path, "r") as f:
            for line in f:
                information = []
                raw_information = line.strip("\n").split(" ")
                for sub_info in raw_information:
                    if sub_info != "":
                        information.append(sub_info)
                information_list.append(information)

        cm_shape = (int(information[0]) + 1, int(information[1]) + 1)
        information_numpy = np.array(information_list)
        dist_map = information_numpy[:, 2].reshape(cm_shape).astype(np.float)
        omega_map = information_numpy[:, 3].reshape(cm_shape).astype(np.float)
        theta_map = information_numpy[:, 4].reshape(cm_shape).astype(np.float)
        phi_map = information_numpy[:, 5].reshape(cm_shape).astype(np.float)

        return dist_map, omega_map, theta_map, phi_map

    def discretize_dist(self, dist_map):
        # build threshold list
        threshold_list = []
        threshold_num = 37
        for i in range(threshold_num):
            threshold_list.append(2 + i * 0.5)

        # discretization
        dist_bin_map = 0
        # index 1 - 36; otherwise index 0 = 0
        for i in range(threshold_num - 1):
            mask = (dist_map >= threshold_list[i]) & (dist_map < threshold_list[i+1])
            dist_bin_map = dist_bin_map + mask * (i + 1)

        # elements on the diagonal line and other disordered values
        #dist_bin_map = dist_bin_map + (dist_map < threshold_list[0]) * (-1)

        return dist_bin_map

    def discretize_omega(self, omega_map, no_contact_mask):
        """
        :param omega_map: -pi, +pi
        :param no_contact_mask: > 20 A.
        :return:
        """
        # convert radian to degree (0-360)
        omega_map = omega_map / np.pi * 0.5
        neg_mask = omega_map < 0
        omega_map[neg_mask] = 0.5 - omega_map[neg_mask]
        omega_map = omega_map * 360
        omega_map[omega_map == 360] = 0

        # build threshold list
        threshold_list = []
        threshold_num = 25
        for i in range(threshold_num):
            threshold_list.append(i * 15)

        # discretization
        omega_bin_map = 0
        for i in range(threshold_num-1):
            mask = (omega_map >= threshold_list[i]) & (omega_map < threshold_list[i+1])
            omega_bin_map = omega_bin_map + mask * (i + 1)
        omega_bin_map[no_contact_mask] = 0

        return omega_bin_map

    def discretize_theta(self, theta_map, no_contact_mask):
        """
        :param theta_map: -pi, +pi
        :param no_contact_mask: > 20 A.
        :return:
        """
        # convert radian to degree (0-360)
        theta_map = theta_map / np.pi * 0.5
        neg_mask = theta_map < 0
        theta_map[neg_mask] = 0.5 - theta_map[neg_mask]
        theta_map = theta_map * 360
        theta_map[theta_map == 360] = 0

        # build threshold list
        threshold_list = []
        threshold_num = 25
        for i in range(threshold_num):
            threshold_list.append(i * 15)

        # discretization
        theta_bin_map = 0
        for i in range(threshold_num - 1):
            mask = (theta_map >= threshold_list[i]) & (theta_map < threshold_list[i + 1])
            theta_bin_map = theta_bin_map + mask * (i + 1)
        theta_bin_map[no_contact_mask] = 0

        return theta_bin_map

    def discretize_phi(self, phi_map, no_contact_mask):
        """
        :param phi_map: -pi, +pi
        :param no_contact_mask: > 20 A.
        :return:
        """
        # convert radian to degree (0-180)
        phi_map = phi_map / np.pi * 180
        phi_map[phi_map == 180] = 0

        # build threshold list
        threshold_list = []
        threshold_num = 13
        for i in range(threshold_num):
            threshold_list.append(i * 15)

        # discretization
        phi_bin_map = 0
        for i in range(threshold_num - 1):
            mask = (phi_map >= threshold_list[i]) & (phi_map < threshold_list[i + 1])
            phi_bin_map = phi_bin_map + mask * (i + 1)
        phi_bin_map[no_contact_mask] = 0

        return phi_bin_map


    def read_pss(self, pss_path):
        rss = []
        with open(pss_path, "r") as f:
            for line in f:
                if ">" in line:
                    continue
                for character in line.strip():
                    if character == "-":  # is that should continue?
                        rss_label = -1
                    else:
                        rss_label = self.class_to_label[character]
                    rss.append(rss_label)

        return rss

    def read_rss(self, rss_path):
        rss = np.load(rss_path)
        return rss

    def read_rna_distance(self, rna_dist_path):
        rna_dist = pd.read_csv(rna_dist_path, header=None, sep="\t").values
        rna_dist = rna_dist[:rna_dist.shape[0], :rna_dist.shape[0]]

        return rna_dist

