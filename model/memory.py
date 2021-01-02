import torch
import numpy as np
import random
import math

WORD_NUM = 35000

class EpisodicMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, buffer=None):
        if buffer is None:
            self.memory = {}
        else:
            self.memory = buffer
            total_keys = len(buffer.keys())
            # convert the keys from np.bytes to np.float32
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, WORD_NUM)

    def update(self):
        total_keys = len(self.memory.keys())
        # convert the keys from np.bytes to np.float32
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, WORD_NUM)
        print("memory updated successfully. keys size:", self.all_keys.shape)

    def get_keys(self, indexList):
        keys = []
        for ids in indexList:
            key = np.zeros(WORD_NUM, dtype = np.float32)
            for num in ids:
                key[num] += 1.0
            keys.append(key)
        return np.array(keys)

    def build(self, inputData, rate = 0.1):
        """
        Sample from inputData to form a memory
        inputData: BoolQDataset defined in data/dataset.py
        memory: {key:value}
        key is a 1d numpy array which can measure the distance between data
        the numbers in key must be np.float32. otherwise it may fail in self.update()
        value is a tuple (content, attn_mask, label)
        in other words, value is tokenized data which can be directly fed into model
        """
        self.memory = {}
        total_keys = len(inputData.input_ids)
        sample_size = round(total_keys * rate)
        indexList = random.sample(range(0, total_keys), sample_size)
        
        for idx in indexList:
            key = self.get_keys([inputData.input_ids[idx]])
            val = (inputData.input_ids[idx], inputData.attention_masks[idx], inputData.answers[idx])
            self.memory.update({key.tobytes(): val})

        self.update()

    def _prepare_batch(self, sample):
        """
        Parameter:
        sample -> list of tuple of experiences
               -> i.e, [(content_1,attn_mask_1,label_1),.....,(content_k,attn_mask_k,label_k)]
        Returns:
        batch -> tuple of list of content,attn_mask,label
              -> i.e, ([content_1,...,content_k],[attn_mask_1,...,attn_mask_k],[label_1,...,label_k])
        """
        contents = []
        attn_masks = []
        labels = []
        # Iterate over experiences
        for content, attn_mask, label in sample:
            # convert the batch elements into torch.LongTensor
            contents.append(content)
            attn_masks.append(attn_mask)
            labels.append(label)

        return (torch.tensor(contents, dtype = torch.long), 
                torch.tensor(attn_masks, dtype = torch.long), 
                torch.tensor(labels, dtype = torch.long))

    def get_neighbours(self, keys, k=32):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []
        # Iterate over all the input keys
        # to find neigbours for each of them
        for key in keys:
            # compute similarity scores based on Euclidean distance metric
            similarity_scores = np.dot(self.all_keys, key.T)
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]
            # converts experiences into batch
            batch = self._prepare_batch(neighbours)
            samples.append(batch)

        return samples