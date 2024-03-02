from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def grid_search(X, y, model_name):
    if model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(random_state=13)
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10, 20, 30]
        }
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(random_state=13)
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 5, 10],
            'min_samples_split': [5, 10, 20, 30]
        }
    elif model_name == "XGBoostRegressor":
        model = XGBRegressor(random_state=13)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    elif model_name == "LightGBM":
        model = LGBMRegressor(random_state=13)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    elif model_name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(random_state=13)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    elif model_name == "AdaBoostRegressor":
        model = AdaBoostRegressor(random_state=13)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    elif model_name == "KNeighborsRegressor":
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': [5, 10, 20, 30],
            'weights': ['uniform', 'distance']
        }
    else:
        raise ValueError("Invalid model name")
    
    # 모델평가는 MAE
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

# 모델평가
def evaluate(model, X, y):
    # 예측값 계산
    y_pred = model.predict(X)
    
    # MAE 계산
    mae = mean_absolute_error(y, y_pred)
    print("MAE:", mae)
    
    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("RMSE:", rmse)

    
# 각각의 모델
def tr_dt(X, y, max_depth=None, min_samples_split=2):
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=13)
    model.fit(X, y)
    return model

def tr_rf(X, y, n_estimators=100, max_depth=None, min_samples_split=2):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=13)
    model.fit(X, y)
    return model

def tr_knn(X, y, n_neighbors=5, weights='uniform'):
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model.fit(X, y)
    return model

def tr_xgb(X, y, n_estimators=50, max_depth=None, learning_rate=0.05):
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=13)
    model.fit(X, y)
    return model

def tr_adaboost(X, y, n_estimators=50, learning_rate=0.1):
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=13)
    model.fit(X, y)
    return model

def tr_lgb(X, y, max_depth=2, learning_rate=0.1, n_estimators=100):
    model = LGBMRegressor(max_depth=max_depth,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators)
    model.fit(X, y)
    return model

def tr_gbm(X, y, max_depth=2, learning_rate=0.1, n_estimators=100):
    model = GradientBoostingRegressor(max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      n_estimators=n_estimators)
    model.fit(X, y)
    return model