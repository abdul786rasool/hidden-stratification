import kaggle

kaggle.api.authenticate()

owner_slug = "fedoriano"  # Replace with the actual owner slug
dataset_slug = "cifar100"  # Replace with the actual dataset slug
dataset_version_number = None  # If you want the latest version, keep it as None

raw_folder = r'C:\Users\Abdul Rasool\OneDrive\Desktop\Ed 1\BTP\hidden-stratification-master\hidden-stratification-master\test_data'

kaggle.api.dataset_download_files('fedesoriano/cifar100',path=raw_folder, unzip=True)
