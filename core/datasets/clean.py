import pandas as pd
from sklearn.model_selection import train_test_split as split


def _clean_translation_dataset(raw_dir, info):
    raw_data_path = raw_dir / "train_pairs.txt"
    raw_data = pd.read_csv(raw_data_path, names=["x", "y"], **info["parse_args"])
    length = raw_data.shape[0]

    all_smiles, targets, is_x, is_y, is_val, is_test = [], [], [], [], [], []

    all_smiles += raw_data.x.tolist()
    targets += raw_data.y.tolist()
    is_x += [True] * length
    is_y += [False] * length
    is_val += [False] * length
    is_test += [False] * length

    all_smiles += raw_data.y.tolist()
    targets += ["*"] * length
    is_x += [False] * length
    is_y += [True] * length
    is_val += [False] * length
    is_test += [False] * length

    raw_data_path = raw_dir / "valid.txt"
    raw_data = pd.read_csv(raw_data_path, names=["smiles"], **info["parse_args"])
    length = raw_data.shape[0]

    all_smiles += raw_data.smiles.tolist()
    targets += ["*"] * length
    is_x += [False] * length
    is_y += [False] * length
    is_val += [True] * length
    is_test += [False] * length

    raw_data_path = raw_dir / "test.txt"
    raw_data = pd.read_csv(raw_data_path, names=["smiles"], **info["parse_args"])
    length = raw_data.shape[0]

    all_smiles += raw_data.smiles.tolist()
    targets += ["*"] * length
    is_x += [False] * length
    is_y += [False] * length
    is_val += [False] * length
    is_test += [True] * length

    return pd.DataFrame({
        "smiles": all_smiles,
        "target": targets,
        "is_x": is_x,
        "is_y": is_y,
        "is_train": [x or y for (x, y) in zip(is_x, is_y)],
        "is_val": is_val,
        "is_test": is_test})


def _fix_consistency(df):
    val_data = df[df.is_val == True]
    test_data = df[df.is_test == True]
    train_data = df[df.is_train == True]

    x_data = train_data[train_data.is_x == True]
    y_data = train_data[train_data.is_y == True]
    y_data = y_data[y_data.smiles.isin(x_data.target)]

    safe_data = pd.concat([x_data, y_data, val_data, test_data])
    return safe_data.reset_index(drop=True)


def _postprocess_translation_dataset(cleaned_data, raw_data):
    cleaned_data["target"] = raw_data.target
    cleaned_data["is_x"] = raw_data.is_x
    cleaned_data["is_y"] = raw_data.is_y
    cleaned_data["is_train"] = raw_data.is_train
    cleaned_data["is_val"] = raw_data.is_val
    cleaned_data["is_test"] = raw_data.is_test
    return _fix_consistency(cleaned_data)


def _assign_splits(df):
    num_samples = df.shape[0]
    train_indices, val_indices = split(range(num_samples), test_size=0.1)
    df["is_train"] = False
    df["is_val"] = False
    df.loc[train_indices, "is_train"] = True
    df.loc[val_indices, "is_val"] = True
    return df


def clean_ZINC(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data = raw_data.replace(r'\n', '', regex=True)
    return raw_data


def postprocess_ZINC(cleaned_data, raw_data):
    return _assign_splits(cleaned_data)


def clean_drd2(raw_dir, info):
    return _clean_translation_dataset(raw_dir, info)


def postprocess_drd2(cleaned_data, raw_data):
    return _postprocess_translation_dataset(cleaned_data, raw_data)


def clean_logp04(raw_dir, info):
    return _clean_translation_dataset(raw_dir, info)


def postprocess_logp04(cleaned_data, raw_data):
    return _postprocess_translation_dataset(cleaned_data, raw_data)


def clean_logp06(raw_dir, info):
    return _clean_translation_dataset(raw_dir, info)


def postprocess_logp06(cleaned_data, raw_data):
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
    return _assign_splits(cleaned_data)


def clean_ChEMBL(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data.columns = ["smiles"]
    return raw_data


def postprocess_CHEMBL(cleaned_data, raw_data):
    return _assign_splits(cleaned_data)


def clean_moses(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    return raw_data


def postprocess_moses(cleaned_data, raw_data):
    return _assign_splits(cleaned_data)


def clean_AID1706(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    return raw_data


def postprocess_AID1706(cleaned_data, raw_data):
    cleaned_data["orig_smiles"] = raw_data.smiles
    cleaned_data["activity"] = raw_data.activity
    return _assign_splits(cleaned_data)
