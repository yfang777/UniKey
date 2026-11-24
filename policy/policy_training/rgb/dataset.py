from lerobot.datasets.lerobot_dataset import LeRobotDataset


class RgbLeRobotDataset(LeRobotDataset):
    def __init__(self, required_columns, *args, **kwargs):
        kwargs["download_videos"] = True
        super().__init__(*args, **kwargs)

        self.required_columns = tuple(required_columns)
        self.padding_columns = (
            tuple(f"{key}_is_pad" for key in self.delta_timestamps)
            if self.delta_timestamps is not None
            else ()
        )
        available_columns = set(self.hf_dataset.column_names) | set(self.meta.video_keys)
        assert set(self.required_columns).issubset(available_columns), (
            f"{self.required_columns} contains columns missing from the RGB dataset"
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        selected = {key: item[key] for key in self.required_columns}
        for key in self.padding_columns:
            selected[key] = item[key]
        selected["task"] = item["task"]
        return selected
