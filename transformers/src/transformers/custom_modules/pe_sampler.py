from torch.nn.modules.loss import _Loss
import torch
import random
from .pe_utils import PeUtils
import numpy as np
from functools import partial
from multiprocessing import Pool
import torch.multiprocessing as mp

class SeqPeMainTaskPosSampler:
    """
    The purpose of this data sampler is to generate the information for positions used to train ViT model.
    E.g., for the encoded images with shape 14x14, we will generate 196=14*14 PEs for training.
    """
    def __init__(self, data_shape, max_sample_position, max_digits, pad_value=0, data_dim=1, add_cls_token_pe=False, device=torch.device("cpu"), use_random_shift=False, random_shift_rate=1, random_shift_downsample=320, default_start_pos=None, start_epoch=0):
        """
        NOTE: explanation for add_cls_token_pe
        add_cls_token_pe=Ture: the PE model will generate the PE of cls.
        add_cls_token_pe=False: we use a special pos embedding to model the pos of cls.

        This is because that seq_pe is hard to represent cls's pos in 2d situation.
        """
        self.use_random_shift = use_random_shift
        self.random_shift_downsample = random_shift_downsample
        self._epoch = -1
        if isinstance(data_shape, tuple):
            flat_data_shape = data_shape[0] * data_shape[1]
        else:
            flat_data_shape = int(data_shape)
        if data_dim == 1:
            # this is only need for pe_type != seq_pe
            max_len = flat_data_shape + 1 if add_cls_token_pe else flat_data_shape
            self.max_sample_position = max_sample_position - max_len
            if self.use_random_shift:
                pos_list = []
                for i in range(self.max_sample_position + 1):
                    pos_list.append(list(range(i, i+max_len)))
                indices = list(range(len(pos_list)))
                random.shuffle(indices)
                if random_shift_downsample > 0:
                    indices = indices[:random_shift_downsample]
                self.indices = indices
                self.pos_list = [pos_list[i] for i in self.indices]
            if default_start_pos is None:
                self.default_pos_list = list(range(max_len))
            else:
                self.default_pos_list = list(range(default_start_pos, default_start_pos+max_len))
        else:
            assert not add_cls_token_pe
            self.max_sample_position = int(np.sqrt(max_sample_position)) - data_shape[0]
            pos_list = []
            if self.use_random_shift:
                for i in range(self.max_sample_position + 1):
                    for j in range(self.max_sample_position + 1):
                        pos_list.append(PeUtils.create_2d_pos_list(i, j, i+data_shape[0], j+data_shape[1]))
                indices = list(range(len(pos_list)))
                random.shuffle(indices)
                if random_shift_downsample > 0:
                    indices = indices[:random_shift_downsample]
                self.indices = indices
                self.pos_list = [pos_list[i] for i in self.indices]
            if default_start_pos is None:
                self.default_pos_list = PeUtils.create_2d_pos_list(0, 0, data_shape[0], data_shape[1])
            else:
                self.default_pos_list = PeUtils.create_2d_pos_list(
                    default_start_pos[0], default_start_pos[1], default_start_pos[0]+data_shape[0], default_start_pos[1]+data_shape[1]
                )

        self.pad_value = pad_value
        self.data_dim = data_dim
        pos_seq_data, pad_mask = [], []
        self.default_pos_seq_data, self.default_pad_mask = PeUtils.prepare_seqpe_data(self.default_pos_list, max_digits, device, pad_value=self.pad_value)
        self.default_pos_ids = torch.tensor(self.default_pos_list, dtype=torch.long, device=device)

        if self.use_random_shift:
            for raw_pos_list in self.pos_list:
                cur_pos_seq_data, cur_pad_mask = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits, device, pad_value=self.pad_value)
                pos_seq_data.append(cur_pos_seq_data)
                pad_mask.append(cur_pad_mask)
        
            self.pos_seq_data = torch.stack(pos_seq_data, dim=0)
            self.pad_mask = torch.stack(pad_mask, dim=0)
            self.pos_ids = torch.tensor(self.pos_list, dtype=torch.long)
            # self.dataset_indices = list(range(len(self.pos_list)))
        self.max_dataset_index = len(self.pos_list) if self.use_random_shift else 1
        self.batch_ptr = 0
        self.random_shift_rate = random_shift_rate
        self.set_epoch(epoch=start_epoch-1)
    
    def set_epoch(self, epoch=-1):
        if epoch < 0:
            self.batch_ptr = int(32 * random.uniform(0, 1))
            self._epoch += 1
        else:
            assert epoch > self._epoch
            for _ in range(self._epoch, epoch):
                self.batch_ptr = int(32 * random.uniform(0, 1))

    def next(self, device=torch.device("cpu"), batch_size=1):
        """
        max_digits: the maximum length of the padded position sequence
        contrastive_size: the number of contrastive positions to be sampled for each pivot position
        """
        if self.batch_ptr >= self.max_dataset_index:
            self.set_epoch()
        random_value = random.uniform(0, 1)
        if self.use_random_shift and random_value < self.random_shift_rate:
            s = min(self.max_dataset_index-batch_size, self.batch_ptr)
            e = s + batch_size
            self.batch_ptr += batch_size
            return {
                "pos_seq_data": self.pos_seq_data[s:e].flatten(0, 1).to(device),
                "pad_mask": self.pad_mask[s:e].flatten(0, 1).to(device),
                "pos_ids": self.pos_ids[s:e].flatten(0, 1).to(device),
                "batch_size": batch_size
            }
        else:
            return {
                "pos_seq_data": self.default_pos_seq_data.to(device),
                "pad_mask": self.default_pad_mask.to(device),
                "pos_ids": self.default_pos_ids.to(device),
                "batch_size": 1
            }


class SeqPeMainTaskPosMpSampler:
    """
    The purpose of this data sampler is to generate the information for positions used to train ViT model.
    E.g., for the encoded images with shape 14x14, we will generate 196=14*14 PEs for training.
    """
    def __init__(self, data_shape, max_sample_position, max_digits, pad_value=0, data_dim=1, add_cls_token_pe=False, device=torch.device("cpu"), use_random_shift=False, random_shift_rate=1, num_worker=1):
        """
        NOTE: explanation for add_cls_token_pe
        add_cls_token_pe=Ture: the PE model will generate the PE of cls.
        add_cls_token_pe=False: we use a special pos embedding to model the pos of cls.

        This is because that seq_pe is hard to represent cls's pos in 2d situation.
        """
        self.num_worker = num_worker
        # self.worker = SeqPeMainTaskPosRayInnerSampler.options(max_concurrency=num_worker, num_cpus=num_worker).remote(
        #     data_shape, max_sample_position, max_digits, pad_value, data_dim, add_cls_token_pe,
        #     torch.device("cpu"), use_random_shift, random_shift_rate
        # )
        
        self.use_random_shift = use_random_shift
        if isinstance(data_shape, tuple):
            flat_data_shape = data_shape[0] * data_shape[1]
        else:
            flat_data_shape = int(data_shape)
        if data_dim == 1:
            # this is only need for pe_type != seq_pe
            max_len = flat_data_shape + 1 if add_cls_token_pe else flat_data_shape
            self.max_sample_position = max_sample_position - max_len
            if self.use_random_shift:
                pos_list = list(range(self.max_sample_position + 1))
                indices = list(range(len(pos_list)))
                random.shuffle(indices)
                self.pos_list = [pos_list[i] for i in indices]
                self.indices = indices
            self.default_pos_list = list(range(max_len))
        else:
            assert not add_cls_token_pe
            self.max_sample_position = int(np.sqrt(max_sample_position)) - data_shape[0]
            pos_list = []
            if self.use_random_shift:
                for i in range(self.max_sample_position + 1):
                    for j in range(self.max_sample_position + 1):
                        pos_list.append((i, j))
                indices = list(range(len(pos_list)))
                random.shuffle(indices)
                self.pos_list = [pos_list[i] for i in indices]
                self.indices = indices
            self.default_pos_list = PeUtils.create_2d_pos_list(0, 0, data_shape[0], data_shape[1])

        self.pad_value = pad_value
        self.data_dim = data_dim
        self.default_pos_seq_data, self.default_pad_mask = PeUtils.prepare_seqpe_data(self.default_pos_list, max_digits, device, pad_value=self.pad_value)
        self.default_pos_ids = torch.tensor(self.default_pos_list, dtype=torch.long, device=device)
        self.random_shift_rate = random_shift_rate
        self.batch_ptr = 0
        self.max_dataset_index = len(self.pos_list) if self.use_random_shift else 1
        # self.process_fn = partial(SeqPeMainTaskPosMpSampler.process, data_shape, max_digits, pad_value)
        self.max_digits = max_digits
        self.data_shape = data_shape
        self.shared_pos_seq_data, self.shared_pad_mask, self.shared_pos_ids = None, None, None

    def set_epoch(self):
        self.batch_ptr = 0
    
    @staticmethod
    def worker_process(sample_idx, pos, shared_pos_seq_data, shared_pad_mask, shared_pos_ids,
                       data_shape, max_digits, pad_value):
        """
        Worker function that computes a sample and writes the result directly into the shared tensors.
        """
        if isinstance(pos, int):
            raw_pos_list = list(range(i, i+data_shape))
        elif isinstance(pos, tuple):
            i, j = pos
            raw_pos_list = PeUtils.create_2d_pos_list(i, j, i+data_shape[0], j+data_shape[1])
        else:
            raise NotImplementedError
        cur_pos_seq_data, cur_pad_mask = PeUtils.prepare_seqpe_data(raw_pos_list, max_digits, pad_value=pad_value)
        pos_ids = torch.tensor(raw_pos_list, dtype=torch.long)

        # sample_data = SeqPeMainTaskPosMpSampler.process(data_shape, max_digits, pad_value, pos)
        # Copy the computed data into the pre-allocated shared memory at the given index
        shared_pos_seq_data[sample_idx].copy_(cur_pos_seq_data)
        shared_pad_mask[sample_idx].copy_(cur_pad_mask)
        shared_pos_ids[sample_idx].copy_(pos_ids)

    def next(self, device=torch.device("cpu"), batch_size=1):
        """
        max_digits: the maximum length of the padded position sequence
        contrastive_size: the number of contrastive positions to be sampled for each pivot position
        """
        if self.batch_ptr >= self.max_dataset_index:
            self.set_epoch()
        if self.use_random_shift and random.uniform(0, 1) < self.random_shift_rate:
            s = min(self.max_dataset_index-batch_size, self.batch_ptr)
            e = s + batch_size
            self.batch_ptr += batch_size
            samples = self.pos_list[s:e]
            if self.shared_pos_seq_data is None:
                self.shared_pos_seq_data = self.default_pos_seq_data.new_empty(
                    (batch_size, *self.default_pos_seq_data.shape)).share_memory_()
                self.shared_pad_mask = self.default_pad_mask.new_empty(
                    (batch_size, *self.default_pad_mask.shape)).share_memory_()
                self.shared_pos_ids = self.default_pos_ids.new_empty(
                    (batch_size, *self.default_pos_ids.shape)).share_memory_()

            # Prepare the arguments for each worker.
            args = [
                (i, pos, self.shared_pos_seq_data, self.shared_pad_mask, self.shared_pos_ids,
                 self.data_shape, self.max_digits, self.pad_value)
                for i, pos in enumerate(samples)
            ]
            # Use a multiprocessing pool (with torch.multiprocessing) to process samples in parallel.
            with Pool(processes=self.num_worker) as pool:
                pool.starmap(SeqPeMainTaskPosMpSampler.worker_process, args)
            return {
                "pos_seq_data": self.shared_pos_seq_data.clone().flatten(0, 1).to(device),
                "pad_mask": self.shared_pad_mask.clone().flatten(0, 1).to(device),
                "pos_ids": self.shared_pos_ids.clone().flatten(0, 1).to(device),
                "batch_size": batch_size
            }
        else:
            return {
                "pos_seq_data": self.default_pos_seq_data.to(device),
                "pad_mask": self.default_pad_mask.to(device),
                "pos_ids": self.default_pos_ids.to(device),
                "batch_size": 1
            }
        

class SeqPeContrstiveDataSampler:
    """
    This part is designed to force the PEs of closer positions have more similar representations in the hidden space.

    E.g., PE of (10, 10) should be closer with that of (9, 9) among a given list of positions [(0, 1), (3, 8), (10, 13), (9, 9), ...]
    This sampler is to sample the positive and negative data. It will be used together with SeqPeContrastiveCriterion 
    in models/pe_criterion.py
    """
    def __init__(self, n_batch_per_epoch, batch_size, max_sample_position, max_digits, contrastive_size, distributional_sample_range=1024, pad_value=0, data_dim=1, debug_mode=False, max_prefetch=32, start_epoch=0, seed=None):
        self.n_batch_per_epoch = n_batch_per_epoch
        self.batch_size = batch_size
        self.max_digits = max_digits
        self.contrastive_size = contrastive_size
        if data_dim == 1:
            self.distributional_sample_range = distributional_sample_range
            self.max_sample_position = max_sample_position # flatten 2d pos into 1d
        else:
            self.distributional_sample_range = int(np.sqrt(distributional_sample_range))
            self.max_sample_position = int(np.sqrt(max_sample_position))
        self.pad_value = pad_value
        self.data_dim = data_dim
        """
        intuitively set,
        50% for fully random sampling the negatives
        50% ~ 80% for sampling negatives with hard negatives
        80% ~ 100% for distributional sampling, i.e., sampling from a small
        range to force the model knowing the tiny difference between close numbers.
        """
        self.method_ratio = 0.4
        self.hard_neg_ratio = 0.25
        self.pivot_data = None
        self.pivot_pad_mask = None
        self.neg_data = None
        self.neg_pad_mask = None
        self.labels = None
        self.batch_ptr = 0
        self.pos_list = []
        self._epoch = mp.Value('i', -1)
        self.start_epoch = start_epoch
        self.seed = seed
        
        # self._prefetch_data() # debug purpose
        ctx = mp.get_context("fork")
        self.data_queue = ctx.Queue(maxsize=max_prefetch)
        self.stop_event = ctx.Event()
        self.process = ctx.Process(target=self._prefetch_data)
        self.process.daemon = True  # Process will close when main program exits
        self.process.start()

    def stop(self):
        """Stop the background prefetching process."""
        self.stop_event.set()
        self.process.join()

    @property
    def epoch(self):
        return self._epoch.value

    def _shuffle(self, indices):
        self.pivot_data = self.pivot_data[indices]
        self.pivot_pad_mask = self.pivot_pad_mask[indices]
        self.neg_data = self.neg_data[indices]
        self.neg_pad_mask = self.neg_pad_mask[indices]
        self.labels = self.labels[indices]

    def set_epoch(self, epoch=-1):
        self.batch_ptr = 0
        if epoch < 0:
            self.init_dataset()
            with self._epoch.get_lock():
                self._epoch.value += 1
        else:
            assert epoch > self._epoch.value
            for _ in range(self._epoch.value, epoch):
                self.init_dataset()
                with self._epoch.get_lock():
                    self._epoch.value += 1
        # print(f"set epoch: {self._epoch.value}")

    def _prefetch_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.set_epoch(epoch=self.start_epoch)

        while True:
            if self.stop_event.is_set():
                break
            if self.batch_ptr >= self.n_batch_per_epoch:
                self.set_epoch()
            s, e = self.batch_ptr * self.batch_size, (self.batch_ptr+1) * self.batch_size
            self.batch_ptr += 1
            data = {
                "pivot_seq_data": self.pivot_data[s:e],
                "neg_seq_data": self.neg_data[s:e].flatten(0, 1),
                "labels": self.labels[s:e],
                "pivot_pad_mask": self.pivot_pad_mask[s:e],
                "neg_pad_mask": self.neg_pad_mask[s:e].flatten(0, 1)
            }
            self.data_queue.put(data)

    def next(self, device=torch.device("cpu")):
        data = self.data_queue.get()
        data = {k: v.to(device) for k, v in data.items()}
        return data

    def init_dataset(self, debug_mode=None):
        n_max, n_batch, bs = self.max_sample_position, self.n_batch_per_epoch, self.batch_size
        if self.data_dim == 1:
            pos_list = list(range(n_max))
        else:
            pos_list = PeUtils.create_2d_pos_list(0, 0, n_max, n_max)
        self.pos_list = pos_list
        if n_batch * bs < len(pos_list):
            n_replica, n_reminder = 1, bs - (len(pos_list) % bs)
            n_reminder = n_reminder % bs
        else:
            n_replica, n_reminder = n_batch * bs // len(pos_list), n_batch * bs % len(pos_list)
        pos_list = pos_list * n_replica + random.sample(pos_list, n_reminder)
        random.shuffle(pos_list)
        pivot_data_list, pivot_pad_mask_list, neg_data_list, neg_pad_mask_list, labels_list = [], [], [], [], []
        n_pos = len(pos_list)
        for i in range(0, n_pos, bs):
            pivot_list = pos_list[i:i + bs]
            pivot_data, pivot_pad_mask, neg_data, neg_pad_mask, labels = self._prepare_batch(pivot_list, debug_mode=debug_mode)
            pivot_data_list.append(pivot_data)
            pivot_pad_mask_list.append(pivot_pad_mask)
            neg_data_list.append(neg_data)
            neg_pad_mask_list.append(neg_pad_mask)
            labels_list.append(labels)
        self.pivot_data = torch.cat(pivot_data_list, dim=0)
        self.pivot_pad_mask = torch.cat(pivot_pad_mask_list, dim=0)
        self.neg_data = torch.cat(neg_data_list, dim=0)
        self.neg_pad_mask = torch.cat(neg_pad_mask_list, dim=0)
        self.labels = torch.cat(labels_list, dim=0)
        self.batch_ptr = 0
        
    def generate_hard_negatives(self, pivot_value, n_hard, min_square_dist=0):
        """
        Generating hard negatives that are similar to the sequence of pivot position. E.g., pivot is 123 --> hard neg 12
        """
        n_max = self.max_sample_position if isinstance(pivot_value, int) else self.max_sample_position**2
        max_digits = self.max_digits
        pivot_value_array = np.array([pivot_value]) if isinstance(pivot_value, int) else np.array(pivot_value)
        def add_digit(input_num_str, num_len):
            if num_len == max_digits:
                return -1
            position = random.randint(0, num_len)
            random_num = random.randint(1, 9) if position == 0 else random.randint(0, 9)
            result = input_num_str[:position] + str(random_num) + input_num_str[position:]
            if int(result) >= n_max:
                return -1
            return result
        
        def delete_digit(input_num_str, num_len):
            if num_len == 1:
                return -1
            position = random.randint(0, num_len - 1)
            while position == 0 and input_num_str[1] == '0':
                position = random.randint(1, num_len - 1)

            return input_num_str[:position] + input_num_str[position + 1:]
        
        def exchange(input_num_str, num_len):
            if num_len == 1 or (num_len == 2 and input_num_str[1] == '0') or (num_len == 2 and input_num_str[0] == input_num_str[1]):
                return -1
            
            position_list = list(range(num_len))
            
            exchange_position = random.sample(position_list, 2)
            exchange_position.sort()
            str_list = list(input_num_str)
            str_list[exchange_position[0]], str_list[exchange_position[1]] = str_list[exchange_position[1]], str_list[exchange_position[0]]
            new_num_str = ''.join(str_list)
            runtime_error_count = 0  
            
            while exchange_position[0] == 0 and input_num_str[exchange_position[1]] == '0' or new_num_str == input_num_str:
                if runtime_error_count == 10:
                    return -1
                exchange_position = random.sample(position_list, 2)
                exchange_position.sort()
                str_list = list(input_num_str)
                str_list[exchange_position[0]], str_list[exchange_position[1]] = str_list[exchange_position[1]], str_list[exchange_position[0]]
                new_num_str = ''.join(str_list)
                runtime_error_count += 1           
            if int(new_num_str) >= n_max:
                return -1
            return new_num_str

        def edit_digit(input_num_str, num_len):
            position = random.randint(0, num_len - 1)
            random_number = random.randint(0, 9)
            
            while position == 0 and random_number == 0 or str(random_number) == input_num_str[position]:
                position = random.randint(0, num_len - 1)
                random_number = random.randint(0, 9)                
            result = input_num_str[: position] + str(random_number) + input_num_str[position + 1:]
            if int(result) >= n_max:
                return -1
            return result

        def generate_hard_negative_number(pivot_value):
            operation_dict = {'1':add_digit,'2': exchange, '3' : edit_digit, '4': delete_digit}
            pivot_value_str = str(pivot_value)
            exe_num = random.randint(1, 3)
            achieved_num = 0
            cur_num_str = str(pivot_value)
            while achieved_num != exe_num:
                mov_select = random.randint(1, 4)
                num_len = len(cur_num_str)
                return_val = operation_dict[f'{mov_select}'](cur_num_str, num_len)
                if return_val != -1 and return_val != pivot_value_str:
                    achieved_num += 1
                    cur_num_str = return_val
            return cur_num_str
        
        def split_num(x):
            x = str(x)
            if len(x) < 2:
                x = (x+'0') if random.uniform(0, 1) < 0.5 else ('0'+x)
            while True:
                k = random.sample(range(1, max_digits+1), 1)[0]
                if 0 < (len(x) - k) <= max_digits:
                    return int(x[:k]), int(x[k:])

        hard_negative_list = []
        from_tuple = False
        if isinstance(pivot_value, tuple):
            from_tuple = True
            pivot_value = "".join([str(v) for v in pivot_value])
            pivot_value = int(pivot_value)

        for _ in range(n_hard):
            hard_negative_num = generate_hard_negative_number(pivot_value)
            hard_negative_num = int(hard_negative_num)
            if from_tuple:
                hard_negative_num = split_num(hard_negative_num)
                hard_negative_array = np.array(hard_negative_num)
            else:
                hard_negative_array = np.array([hard_negative_num])
            dist = np.square(pivot_value_array - hard_negative_array).sum()
            if dist > min_square_dist:
                hard_negative_list.append(hard_negative_num)
        return hard_negative_list

    def distributional_sample(self, pivot_value, n_sample):
        """
        This sample method is to ensure that the model can distinguish the difference between pivot position and a list of close positions.
        E.g., pivot 123 <-> neg list [1, 3, 5, 7]
        """
        if self.data_dim == 1:
            start_point = int((self.max_sample_position-self.distributional_sample_range) * random.uniform(0, 1))
            sample_range = max(self.distributional_sample_range, 5 * n_sample)
            end_point = min(sample_range + start_point, self.max_sample_position)
            pos_list = list(range(start_point, end_point))
            # is_continues_sampling = random.uniform(0, 1) < 1/3
            # sample_range = self.distributional_sample_range
            # max_range = self.max_sample_position
            # is_right = random.uniform(0, 1) > pivot_value / max_range
            # if not is_continues_sampling:
            #     normalized_number = np.exp(-np.random.exponential(scale=1.0))
            #     length = random.randint(sample_range - 50, sample_range + 50)
            #     if is_right:
            #         start_point = max(int(normalized_number * (max_range-pivot_value)), 1) + pivot_value
            #         end_point = min(start_point + length, max_range)
            #     else:
            #         start_point = int((1 - normalized_number) * pivot_value)
            #         end_point = min(start_point + length, pivot_value)
            # else:
            #     # continues number
            #     if is_right:
            #         start_point = random.randint(min(pivot_value, max_range - n_sample - 1), max_range - n_sample)
            #     else:
            #         start_point = random.randint(0, min(pivot_value, max_range - n_sample))
            #     end_point = start_point + n_sample
            # pos_list = list(range(start_point, end_point))
        elif self.data_dim == 2:
            n_max, n_range = self.max_sample_position, self.distributional_sample_range
            n_range = max(n_range, int(np.sqrt(5 * n_sample)))
            p_s = int((n_max-n_range)* random.uniform(0, 1)), int((n_max-n_range)* random.uniform(0, 1))
            p_e = min(n_range + p_s[0], n_max), min(n_range + p_s[1], n_max)
            pos_list = PeUtils.create_2d_pos_list(p_s[0], p_s[1], p_e[0], p_e[1])
        else:
            raise NotImplementedError
        if len(pos_list) <= n_sample:
            random_sample_list =  pos_list
            # print("len(pos_list) < n_sample", len(pos_list), n_sample)
        else:
            random_sample_list = random.sample(pos_list, n_sample)
        return random_sample_list
    
    def _dist_fn(self, pivot_list, neg_list, batch_size, contrastive_size, device):
        if self.data_dim == 1:
            pivot_raw_data = torch.tensor(pivot_list, dtype=torch.float, device=device)[:, None, None]
        else:
            pivot_raw_data = torch.tensor(pivot_list, dtype=torch.float, device=device)[:, None, :]
        neg_raw_data = torch.tensor(neg_list, dtype=torch.float, device=device).reshape(batch_size, contrastive_size, -1)
        dist = torch.sum((neg_raw_data-pivot_raw_data) ** 2, dim=-1)
        return dist

    def _prepare_batch(self, pivot_list, device=torch.device("cpu"), debug_mode=None):
        """
        max_digits: the maximum length of the padded position sequence
        batch_size: the number of pivot positions to be sampled
        contrastive_size: the number of contrastive positions to be sampled for each pivot position
        """
        contrastive_size, max_digits, batch_size = self.contrastive_size, self.max_digits, self.batch_size
        batch_negative_list = []
        n_hard = int(contrastive_size * self.hard_neg_ratio)
        pivot_data, pivot_pad_mask = PeUtils.prepare_seqpe_data(pivot_list, max_digits, device, pad_value=self.pad_value)

        simple_negative_list = []
        for i in range(batch_size):
            simple_negative_list.append(random.sample(self.pos_list, contrastive_size+1))
        dist = self._dist_fn(pivot_list, simple_negative_list, batch_size, contrastive_size+1, device)

        for i in range(batch_size):
            if debug_mode is not None:
                is_dist_sample = debug_mode
            else:
                is_dist_sample = random.uniform(0, 1) > self.method_ratio
            if not is_dist_sample:
                min_square_dist = dist[i, :-n_hard].min().item()
                special_negative_list = self.generate_hard_negatives(pivot_list[i], n_hard, min_square_dist=min_square_dist)
            else:
                special_negative_list = self.distributional_sample(pivot_list[i], contrastive_size)
            merged_negative_list = []
            visited = {pivot_list[i]}
            for pos in special_negative_list + simple_negative_list[i]:
                if pos not in visited:
                    merged_negative_list.append(pos)
                    visited.add(pos)
            batch_negative_list += merged_negative_list[:contrastive_size]
        # if self.pad_value == 0:
        neg_data, neg_pad_mask = PeUtils.prepare_seqpe_data(batch_negative_list, max_digits, device, pad_value=self.pad_value)
        scores = self._dist_fn(pivot_list, batch_negative_list, batch_size, contrastive_size, device=device)
        # print(scores.min(dim=-1))
        labels = torch.argmin(scores, dim=-1)
        neg_data = neg_data.reshape(len(pivot_list), -1, *neg_data.shape[1:])
        neg_pad_mask = neg_pad_mask.reshape(len(pivot_list), -1, *neg_pad_mask.shape[1:])
        return pivot_data, pivot_pad_mask, neg_data, neg_pad_mask, labels


class SeqPeTransferDataSampler:
    """
    The purpose of this sampler is to generate data to align
    PEs of [i, i+k) positions with those of [0, k) positions, where [a, b) is the range of positions.

    This module is designed to improve the extrapolation ability of the PE.
    This sampler will be used togeter with SeqPeTransferCriterion in models/pe_criterion.py
    """
    def __init__(self, data_shape, n_batch_per_epoch, batch_size, max_sample_position, max_digits, transfer_size, pad_value=0, data_dim=1, start_epoch=0):
        """
        NOTE: the original max_sample_position and transfer_size was given in 1d form. Thus, if data_dim==2,
        we should use their squre root.
        """
        self._epoch = -1
        self.n_batch_per_epoch = n_batch_per_epoch
        self.batch_ptr = 0
        self.pos_seq_data = None
        self.pad_mask = None
        self.pivot_indices = None
        self.max_digits = max_digits
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.transfer_size = transfer_size
        if data_dim == 1:
            self.max_sample_position = max_sample_position # flatten 2d pos into 1d
        else:
            self.max_sample_position = int(np.sqrt(max_sample_position))
        
        self.start_epoch = start_epoch
        self.pad_value = pad_value
        self.data_dim = data_dim
        self.set_epoch(epoch=start_epoch)
    
    def _shuffle(self, indices):
        self.pos_seq_data = self.pos_seq_data[indices]
        self.pad_mask = self.pad_mask[indices]
        self.pivot_indices = self.pivot_indices[indices]
    
    def set_epoch(self, epoch=-1):
        self.batch_ptr = 0
        if epoch < 0:
            self.init_dataset()
            self._epoch += 1
        else:
            assert epoch > self._epoch
            for _ in range(self._epoch, epoch):
                self.init_dataset()
                self._epoch += 1
        
    def init_dataset(self,):
        n_max, n_batch, bs = self.max_sample_position, self.n_batch_per_epoch, self.batch_size
        ds = self.data_shape
        if self.data_dim == 1:
            if isinstance(ds, int):
                pos_list = list(range(ds // 4, n_max-ds))
            else:
                pos_list = list(range(ds[0] * ds[1] // 4, n_max-ds[0] * ds[1]))
        else:
            pos_list = PeUtils.create_2d_pos_list(0, 0, n_max-ds[0], n_max-ds[1])
            pos_list = [p for p in pos_list if p[0] >= ds[0] or p[1] >= ds[1]]
        if n_batch * bs < len(pos_list):
            n_replica, n_reminder = 1, bs - (len(pos_list) % bs)
            n_reminder = n_reminder % bs
        else:
            n_replica, n_reminder = n_batch * bs // len(pos_list), n_batch * bs % len(pos_list)
        pos_list = pos_list * n_replica + random.sample(pos_list, n_reminder)
        random.shuffle(pos_list)
        pivot_indices_list, pos_data_list, pad_mask_list = [], [], []
        n_pos = len(pos_list)
        for i in range(0, n_pos, bs):
            start_points = pos_list[i:i + bs]
            pivot_indices, sampled_pos_seq_data, sampled_pad_mask = self._prepare_batch(start_points)
            pos_data_list.append(sampled_pos_seq_data)
            pad_mask_list.append(sampled_pad_mask)
            pivot_indices_list.append(pivot_indices)
        self.pos_seq_data = torch.cat(pos_data_list, dim=0)
        self.pad_mask = torch.cat(pad_mask_list, dim=0)
        self.pivot_indices = torch.cat(pivot_indices_list, dim=0)
        self.batch_ptr = 0

    def next(self, device=torch.device("cpu")):
        if self.batch_ptr >= self.n_batch_per_epoch:
            self.set_epoch()
        s, e = self.batch_ptr * self.batch_size, (self.batch_ptr+1) * self.batch_size
        self.batch_ptr += 1
        return {
            "pivot_indices": self.pivot_indices[s:e].flatten().to(device),
            "pos_seq_data": self.pos_seq_data[s:e].flatten(0, 1).to(device),
            "pad_mask": self.pad_mask[s:e].flatten(0, 1).to(device),
        }

    def _prepare_batch(self, start_points, device=torch.device("cpu")):
        """
        max_digits: the maximum length of the padded position sequence
        batch_size: the number of pivot positions to be sampled
        transfer_size: the number of positions to be calculate the transfer loss
        """
        bs = len(start_points)
        n_t = self.transfer_size
        ds = self.data_shape
        max_digits = self.max_digits
        batch_pivot_indices, sampled_pos_list = [], []
        if isinstance(ds, int):
            orig_indices = list(range(ds))
        else:
            orig_indices = list(range(ds[0] * ds[1]))
        for p in start_points:
            if self.data_dim == 1:
                if isinstance(ds, int):
                    target_pos_list = list(range(p, p + ds))
                else:
                    target_pos_list = list(range(p, p + ds[0] * ds[1]))
            else:
                target_pos_list = PeUtils.create_2d_pos_list(p[0], p[1], p[0]+ds[0], p[1]+ds[1])
            sampled_indices = random.sample(orig_indices, n_t)
            sampled_pos_list += [target_pos_list[i] for i in sampled_indices]
            batch_pivot_indices.append(sampled_indices)

        sampled_pos_seq_data, sampled_pad_mask = PeUtils.prepare_seqpe_data(sampled_pos_list, max_digits, device, pad_value=self.pad_value)
        sampled_pos_seq_data = sampled_pos_seq_data.reshape(bs, -1, *sampled_pos_seq_data.shape[1:])
        sampled_pad_mask = sampled_pad_mask.reshape(bs, -1, *sampled_pad_mask.shape[1:])
        pivot_indices = torch.tensor(batch_pivot_indices, dtype=torch.long, device=device)
        return pivot_indices, sampled_pos_seq_data, sampled_pad_mask


def _test_transfer_sampler(data_dim=1, n_iter=3):
    max_sample_position = 10000
    transfer_size = 16
    data_shape = (14, 14)
    batch_size = 4
    n_batch_per_epoch = 1

    if data_dim == 1:
        max_digits = PeUtils.get_digit_num(max_sample_position)
    else:
        max_digits = PeUtils.get_digit_num(int(np.sqrt(max_sample_position)))
    pe_trans_sampler = SeqPeTransferDataSampler(
        data_shape, n_batch_per_epoch=n_batch_per_epoch, batch_size=batch_size, max_sample_position=max_sample_position,
        max_digits=max_digits, transfer_size=transfer_size, data_dim=data_dim
    )
    print(pe_trans_sampler.pos_seq_data.size())
    print(pe_trans_sampler.pad_mask.size())
    print(pe_trans_sampler.pivot_indices.size())
    print(f"_test_transfer_sampler | max_sample_position={max_sample_position} | batch_size={batch_size} | transfer_size={transfer_size} | data_dim={data_dim}")
    indices = random.sample(range(pe_trans_sampler.pos_seq_data.size(0)), n_iter)
    for i in indices:
        print(pe_trans_sampler.pivot_indices[i].size())
        print(pe_trans_sampler.pivot_indices[i])
        print(pe_trans_sampler.pos_seq_data[i].size())
        print(pe_trans_sampler.pos_seq_data[i])
        print("*" * 50)
    sample = pe_trans_sampler.next()
    print(sample['pivot_indices'].size())
    print(sample['pos_seq_data'].size())
    print(sample['pad_mask'].size())
    print(sample['pivot_indices'][-transfer_size:])
    print(sample['pos_seq_data'][-transfer_size:])
    print(sample['pad_mask'][-transfer_size:])


def _test_contrstive_sampler(data_dim=1, debug_mode=1, n_iter=3):
    max_sample_position = 10000
    batch_size = 4
    contrastive_size = 64
    n_batch_per_epoch = 10
    if data_dim == 1:
        max_digits = PeUtils.get_digit_num(max_sample_position)
    else:
        max_digits = PeUtils.get_digit_num(int(np.sqrt(max_sample_position)))
    pe_ct_sampler = SeqPeContrstiveDataSampler(
        n_batch_per_epoch, batch_size, max_sample_position, max_digits, contrastive_size, distributional_sample_range=196, pad_value=0, data_dim=data_dim
    )
    print(pe_ct_sampler.pivot_data.size())
    print(pe_ct_sampler.neg_data.size())
    print(pe_ct_sampler.labels.size())

    if data_dim == 1:
        pivot_values =[2819, 28, 2010, 540, 3854]
    else:
        pivot_values =[(28, 19), (28, 0), (20, 10), (5, 40), (38, 54)]
    for pivot_value in pivot_values:
        print(f"_test_contrstive_sampler.generate_hard_negatives | max_sample_position={max_sample_position} | batch_size={batch_size} | contrastive_size={contrastive_size} | data_dim={data_dim} | pivot_value={pivot_value}")
        ans = pe_ct_sampler.generate_hard_negatives(pivot_value, 10)
        print(ans)
    
    print("*" * 50 + "\n" + "*" * 50)
    pe_ct_sampler.init_dataset(debug_mode=debug_mode)
    indices = random.sample(range(pe_ct_sampler.pivot_data.size(0)), 3)
    for i in indices:
        print("*" * 50 + f"mode={debug_mode} | iter {i}" + "*" * 50)
        s, e = i*contrastive_size, (i+1)*contrastive_size
        print(pe_ct_sampler.pivot_data[i])
        labels = pe_ct_sampler.labels[i]
        print(labels)
        print(pe_ct_sampler.neg_data[s+labels])
        print(pe_ct_sampler.neg_data[s:e])
        print("*" * 50)


def _test_contrstive_sampler_next(data_dim=1):
    max_sample_position = 10000
    batch_size = 32
    contrastive_size = 64
    n_batch_per_epoch = 625
    if data_dim == 1:
        max_digits = PeUtils.get_digit_num(max_sample_position)
    else:
        max_digits = PeUtils.get_digit_num(int(np.sqrt(max_sample_position)))
    pe_ct_sampler = SeqPeContrstiveDataSampler(
        n_batch_per_epoch, batch_size, max_sample_position, max_digits, contrastive_size, distributional_sample_range=196, pad_value=0, data_dim=data_dim, max_prefetch=32, start_epoch=2, seed=10086
    )
    # sample = pe_ct_sampler.next()
    # print(sample['pivot_seq_data'].size())
    # print(sample['labels'].size())
    # print(sample['neg_seq_data'].size())
    # print(sample['pivot_seq_data'][-1:])
    # print(sample['labels'][-1], sample['neg_seq_data'][-contrastive_size+sample['labels'][-1]:])
    # print(sample['neg_seq_data'][-contrastive_size:]) 

    n = 0
    while True:
        data = pe_ct_sampler.next()
        # if pe_ct_sampler.epoch == 2:
        #     import pdb; pdb.set_trace()
        n += 1
        if n > 2000:
            break
        # if (n % n_batch_per_epoch) == 0:
        #     print(pe_ct_sampler.epoch, n)


def _test_SeqPeMainTaskPosRaySampler():
    import time
    # N=100
    batch_size = 32
    # ray.init(num_cpus=4)
    random.seed(10086)
    torch.manual_seed(10086)
    t_s = time.time()
    old_sampler = SeqPeMainTaskPosSampler(
        data_shape=(14, 14), max_sample_position=10000, max_digits=2,
        pad_value=0, data_dim=2, add_cls_token_pe=False, device=torch.device("cpu"), use_random_shift=True,
        random_shift_rate=1
    )
    n_batch = (len(old_sampler.indices)+batch_size-1) // batch_size
    old_batch_list = [old_sampler.next(batch_size=batch_size) for _ in range(n_batch)]
    print(f"Old sampler time consume {time.time()-t_s:.3f}s")
    random.seed(10086)
    torch.manual_seed(10086)
    sampler = SeqPeMainTaskPosMpSampler(
        data_shape=(14, 14), max_sample_position=10000, max_digits=2,
        pad_value=0, data_dim=2, add_cls_token_pe=False, device=torch.device("cpu"), use_random_shift=True,
        random_shift_rate=1, num_worker=4
    )
    t_s = time.time()
    batch_list = [sampler.next(batch_size=batch_size) for _ in range(n_batch)]
    print(f"Ray sampler time consume {time.time()-t_s:.3f}s")
    print(sum([(batch_list[i]['pos_seq_data']-old_batch_list[i]['pos_seq_data']).sum() for i in range(n_batch)]))


if __name__ == "__main__":
    random.seed(10086)
    torch.manual_seed(10086)
    # _test_SeqPeMainTaskPosRaySampler()
    # _test_transfer_sampler(data_dim=1)
    # _test_transfer_sampler(data_dim=2)

    # _test_contrstive_sampler(data_dim=1, debug_mode=0, n_iter=1)
    # _test_contrstive_sampler(data_dim=1, debug_mode=1, n_iter=1)
    # _test_contrstive_sampler(data_dim=1, debug_mode=2, n_iter=1)

    # _test_contrstive_sampler(data_dim=2, debug_mode=0, n_iter=1)
    # _test_contrstive_sampler(data_dim=2, debug_mode=1, n_iter=1)
    # _test_contrstive_sampler(data_dim=2, debug_mode=2, n_iter=1)

    # _test_contrstive_sampler_next(data_dim=1)
    _test_contrstive_sampler_next(data_dim=2)