import math

import torch

from policy.policy_training.rgb.dataset import RgbLeRobotDataset


class EpisodeSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class CombinedRgbLeRobotDatasetV4(torch.utils.data.Dataset):
    def __init__(self, required_columns, recovery_dataset_path=None,
                 new_recovery_episode_count=0, **dataset_kwargs):
        self.base_dataset = RgbLeRobotDataset(
            required_columns=required_columns,
            **dataset_kwargs,
        )
        self.old_recovery_dataset = None
        self.new_recovery_dataset = None

        if recovery_dataset_path is not None:
            recovery_kwargs = {
                **dataset_kwargs,
                "repo_id": f"{dataset_kwargs['repo_id']}_recovery",
                "root": recovery_dataset_path,
            }
            recovery_dataset = RgbLeRobotDataset(
                required_columns=required_columns,
                **recovery_kwargs,
            )

            if new_recovery_episode_count > 0:
                episode_indices = list(recovery_dataset.hf_dataset["episode_index"])
                last_episode = max(episode_indices)
                first_new_episode = last_episode - int(new_recovery_episode_count) + 1
                old_indices = [
                    idx for idx, episode_idx in enumerate(episode_indices)
                    if episode_idx < first_new_episode
                ]
                new_indices = [
                    idx for idx, episode_idx in enumerate(episode_indices)
                    if episode_idx >= first_new_episode
                ]
                self.old_recovery_dataset = EpisodeSubsetDataset(
                    recovery_dataset,
                    old_indices,
                )
                self.new_recovery_dataset = EpisodeSubsetDataset(
                    recovery_dataset,
                    new_indices,
                )
            else:
                self.old_recovery_dataset = recovery_dataset

        self.sources = []
        self.source_ranges = {}
        offline_datasets = [self.base_dataset]
        if self.old_recovery_dataset is not None and len(self.old_recovery_dataset) > 0:
            offline_datasets.append(self.old_recovery_dataset)
        self.offline_dataset = torch.utils.data.ConcatDataset(offline_datasets)

        self._add_source("offline", self.offline_dataset)
        self._add_source("new_recovery", self.new_recovery_dataset)

        print(f"[CombinedRgbLeRobotDatasetV4] Base length: {len(self.base_dataset)}")
        print(
            "[CombinedRgbLeRobotDatasetV4] Recovery length: "
            f"{0 if self.old_recovery_dataset is None else len(self.old_recovery_dataset)}"
        )
        print(
            "[CombinedRgbLeRobotDatasetV4] Offline length: "
            f"{self._source_len('offline')}"
        )
        print(
            "[CombinedRgbLeRobotDatasetV4] New recovery length: "
            f"{self._source_len('new_recovery')}"
        )

    def _add_source(self, name, dataset):
        if dataset is None or len(dataset) == 0:
            return
        start = len(self)
        self.sources.append((name, dataset))
        self.source_ranges[name] = (start, start + len(dataset))

    def _source_len(self, name):
        if name not in self.source_ranges:
            return 0
        start, end = self.source_ranges[name]
        return end - start

    def __len__(self):
        return sum(len(dataset) for _, dataset in self.sources)

    def __getitem__(self, idx):
        for _, dataset in self.sources:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError(idx)

    def source_indices(self, name):
        start, end = self.source_ranges[name]
        return range(start, end)


class RatioBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, ratios, seed):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        self.source_indices = {
            name: list(dataset.source_indices(name))
            for name in ratios
            if ratios[name] > 0 and name in dataset.source_ranges
        }
        active_total = sum(float(ratios[name]) for name in self.source_indices)
        normalized = {
            name: float(ratios[name]) / active_total
            for name in self.source_indices
        }

        raw_counts = {
            name: normalized[name] * self.batch_size
            for name in self.source_indices
        }
        self.batch_counts = {
            name: int(math.floor(raw_counts[name]))
            for name in self.source_indices
        }
        remaining = self.batch_size - sum(self.batch_counts.values())
        for name, _ in sorted(
            raw_counts.items(),
            key=lambda item: item[1] - math.floor(item[1]),
            reverse=True,
        )[:remaining]:
            self.batch_counts[name] += 1

        print(f"[RatioBatchSampler] Batch counts: {self.batch_counts}")

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(len(self)):
            batch = []
            for name, count in self.batch_counts.items():
                indices = self.source_indices[name]
                draws = torch.randint(
                    low=0,
                    high=len(indices),
                    size=(count,),
                    generator=generator,
                )
                batch.extend(indices[int(i)] for i in draws)

            order = torch.randperm(len(batch), generator=generator)
            yield [batch[int(i)] for i in order]
