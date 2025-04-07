def open_data(path:str) -> 'pd.DataFrame':
    "Maps file extention to loader and calls it"
    ext2loader = {
        '.pickle':lambda p: pd.read_pickle(p),
        '.csv':pd.read_csv,
    }
    _, extention = os.path.splitext(path)
    return ext2loader[extention](path)