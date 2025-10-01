from use_cases.x_ray.bone_fractured.datasets.loader import BoneFracturedDatasetLoader
import os


def test_bone_fractured_dataset_loader():
    loader = BoneFracturedDatasetLoader()
    assert loader.dataset_name == "foyez767/x-ray-images-of-fractured-and-healthy-bones"
    assert loader.download_path == "data/raw"
    loader.download_data()
    assert os.path.exists(loader.download_path)
    data = loader.load_data()
    assert isinstance(data, list)
    assert isinstance(data[0], dict)
    assert "file_path" in data[0]
    assert "sub_dataset" in data[0]
    assert "label" in data[0]
    assert len(data) > 0
