from abc import ABC, abstractmethod
import os
import kaggle


class BaseLoader(ABC):

    @abstractmethod
    def download_data(self, *args, **kwargs): ...

    @abstractmethod
    def load_data(self, *args, **kwargs): ...

    @abstractmethod
    def get_item(self, index: int | str, *args, **kwargs): ...

    @abstractmethod
    def get_batch(self, indices: list[int | str], *args, **kwargs): ...


class KaggleLoader(BaseLoader):
    def __init__(
        self, dataset_name: str, download_path: str = os.path.join("data", "raw")
    ):
        """Initialize the KaggleLoader with dataset name and download path."""
        # chmod 600 ~/.kaggle/kaggle.json
        super().__init__()
        self.download_path = download_path
        self.dataset_name = dataset_name

    def download_data(self, *args, **kwargs):
        kaggle.api.dataset_download_files(
            self.dataset_name,
            path=self.download_path,
            unzip=True,
        )
