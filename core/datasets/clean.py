import pandas as pd


def _clean_translation_dataset(raw_dir, info):
    raw_data_path = raw_dir / "train_pairs.txt"
    raw_data = pd.read_csv(raw_data_path, names=["x", "y"], **info["parse_args"])
    
    all_smiles, is_x, is_y, is_valid, is_test = [], [], [], [], []
    
    all_smiles += raw_data.x.tolist()
    is_x += [1] * raw_data.shape[0]
    is_y += [0] * raw_data.shape[0]
    is_valid += [0] * raw_data.shape[0]
    is_test += [0] * raw_data.shape[0]
    
    all_smiles += raw_data.y.tolist()
    is_x += [0] * raw_data.shape[0]
    is_y += [1] * raw_data.shape[0]
    is_valid += [0] * raw_data.shape[0]
    is_test += [0] * raw_data.shape[0]
    
    raw_data_path = raw_dir / "valid.txt"
    raw_data = pd.read_csv(raw_data_path, names=["smiles"], **info["parse_args"])
    
    all_smiles += raw_data.smiles.tolist()
    is_x += [0] * raw_data.shape[0]
    is_y += [0] * raw_data.shape[0]
    is_valid += [1] * raw_data.shape[0]
    is_test += [0] * raw_data.shape[0]
    
    raw_data_path = raw_dir / "test.txt"
    raw_data = pd.read_csv(raw_data_path, names=["smiles"], **info["parse_args"])
    
    all_smiles += raw_data.smiles.tolist()
    is_x += [0] * raw_data.shape[0]
    is_y += [0] * raw_data.shape[0]
    is_valid += [0] * raw_data.shape[0]
    is_test += [1] * raw_data.shape[0]
    
    return pd.DataFrame({"smiles": all_smiles, "is_x": is_x, "is_y": is_y, "is_valid": is_valid, "is_test": is_test})


def _postprocess_translation_dataset(cleaned_data, raw_data):
    cleaned_data["is_x"] = raw_data.is_x
    cleaned_data["is_y"] = raw_data.is_y
    cleaned_data["is_valid"] = raw_data.is_valid
    cleaned_data["is_test"] = raw_data.is_test
    return cleaned_data


def clean_ZINC(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data = raw_data.replace(r'\n', '', regex=True)
    return raw_data


def postprocess_ZINC(cleaned_data, raw_data):
    return cleaned_data


def clean_drd2(raw_dir, info):
    return _clean_translation_dataset(raw_dir, info)


def postprocess_drd2(cleaned_data, raw_data):
    return _postprocess_translation_dataset(cleaned_data, raw_data)


def clean_logp4(raw_dir, info):
    return _clean_translation_dataset(raw_dir, info)


def postprocess_logp4(cleaned_data, raw_data):
    return _postprocess_translation_dataset(cleaned_data, raw_data)


def clean_logp6(raw_dir, info):
    return _clean_translation_dataset(raw_dir, info)


def postprocess_logp6(cleaned_data, raw_data):
    return _postprocess_translation_dataset(cleaned_data, raw_data)


def clean_qed(raw_dir, info):
    return _clean_translation_dataset(raw_dir, info)


def postprocess_qed(cleaned_data, raw_data):
    return _postprocess_translation_dataset(cleaned_data, raw_data)


def clean_MolPort(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data.columns = ["smiles", "molPortID", "url"]
    return raw_data


def postprocess_MolPort(cleaned_data, raw_data):
    return cleaned_data


def clean_ChEMBL(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data.columns = ["smiles"]
    return raw_data


def postprocess_CHEMBL(cleaned_data, raw_data):
    return cleaned_data


def clean_moses(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    return raw_data


def postprocess_moses(cleaned_data, raw_data):
    return cleaned_data


def clean_AID1706(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    return raw_data


def postprocess_AID1706(cleaned_data, raw_data):
    cleaned_data["orig_smiles"] = raw_data.smiles
    cleaned_data["activity"] = raw_data.activity
    return cleaned_data