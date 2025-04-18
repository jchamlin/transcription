from ctranslate2_utils import get_available_devices, get_available_compute_types as get_ct2_compute_types
from torch_utils import get_available_compute_types as get_torch_compute_types

def main():
    print("CTranslate2 Devices and Compute Types:")
    for device in get_available_devices():
        compute_types = get_ct2_compute_types(device)
        print(f"✅ {device}: {compute_types}")

    print("\nTorch Devices and Compute Types:")
    for device in get_available_devices():
        compute_types = get_torch_compute_types(device)
        print(f"✅ {device}: {compute_types}")

if __name__ == "__main__":
    main()
