"""Transformadores custom del pipeline de riesgo crediticio. Vive en un módulo
propio (en vez de __main__) para que joblib pueda deserializar el modelo
exportado tanto en el proceso de entrenamiento como en el de la API."""
from sklearn.base import BaseEstimator, TransformerMixin


class WinsorizerP99(BaseEstimator, TransformerMixin):
    def __init__(self, columna="utilizacion"):
        self.columna = columna
        self.limite_superior_ = None

    def fit(self, X, y=None):
        self.limite_superior_ = X[self.columna].quantile(0.99)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columna] = X[self.columna].clip(upper=self.limite_superior_)
        return X
