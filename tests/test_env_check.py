from hasar.env_check import check_environment


def test_check_environment_keys():
    info = check_environment()
    assert set(info) >= {"python", "torch", "ultralytics", "cuda_available"}
    assert isinstance(info["cuda_available"], bool)