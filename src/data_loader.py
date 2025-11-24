import pandas as pd
from .config import DATA_RAW
from .preprocessing import clean_diabetes_dataset


def load_diabetes_data(path: str | None = None, clean: bool = False, save_clean_path: str | None = None) -> pd.DataFrame:
    """
    Carga el dataset de diabetes, con opción de aplicar limpieza.

    Parámetros:
    -----------
    path : str | None
        Ruta al CSV. Si no se especifica, usa DATA_RAW.
    clean : bool
        Si True, aplica la función clean_diabetes_dataset.
    save_clean_path : str | None
        Si se especifica, guarda el dataset limpio en esa ruta.

    Retorna:
    --------
    pd.DataFrame: Dataset original o limpio según clean=True/False.
    """

    csv_path = path if path is not None else DATA_RAW
    df = pd.read_csv(csv_path)

    if clean:
        df = clean_diabetes_dataset(df, save_path=save_clean_path)

    return df
