from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score , classification_report
import xgboost as xgb
from xgboost import XGBClassifier

# Load dataset
data = load_breast_cancer()
X,y = data.data,data.target

#split data   
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# display d info 
print("features:",data.feature_names)
print("Classes:",data.target_names)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# train xg boost

params = {
  'objective':'binary:logistic',
  'max_depth': 3,
  'learning_rate': 0.1,
  'eval_metric': 'logloss',
}

xgb_model = xgb.train(params, dtrain, num_boost_round=100)

#predict 

y_pred = (xgb_model.predict(dtest)>0.5).astype(int)

# evaluate performance 
accuracy = accuracy_score(y_test,y_pred)

print("XGBoost Model Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

# hyper parameter tuning 

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

#initialize xgboost classifier 
xgb_clf = XGBClassifier(eval_metric='logloss',random_state=42)

#perform grid search  with cross validation 
grid_search = GridSearchCV(estimator=xgb_clf,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1)

grid_search.fit(X_train,y_train)

#display best parameters and best score 
print("Best Parameters:",grid_search.best_params_)
print("Best Score:",grid_search.best_score_)


#train gradient boosting 
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train,y_train)
y_pred_gb = gb_model.predict(X_test)

# perofrmance evaluation

accuracy_gb = accuracy_score(y_test,y_pred_gb)
print("Gradient Boosting Model Accuracy:",accuracy_gb)


