import pandas
from datetime import datetime
import numpy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
from plots import plot_3d

data = pandas.read_csv('creditcard.csv')

# Imbalanced dataset so will need to resample
print(data['Class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data.Class, test_size=0.20,
                                                    random_state=42)

print(y_train.value_counts()[1] / len(y_train))
print(y_test.value_counts()[1] / len(y_test))
# Under Sampling to deal with imbalance
rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')
X_usampled, y_usampled = rus.fit_sample(X_train, y_train)


# Over Sampling to deal with imbalance
sm = SMOTE(sampling_strategy='minority', random_state=7)

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=X_osampled['Time'], y=X_osampled['Amount'], mode='markers'))
# fig.show()


# Cross Val function to do GridSearchCV
def cross_validation(model, params, X_train, Y_train):
    f1_scorer = make_scorer(f1_score, pos_label=1)
    gs = GridSearchCV(model, params, cv=10, scoring=f1_scorer)
    gs.fit(X_train, Y_train)
    print('F1 Score: ')
    print(gs.best_score_)
    print(gs.best_estimator_)
    return gs.best_estimator_


# Logistic Regression Under
lR = LogisticRegression()
parameters = [{}]
start = datetime.now()
lRuEstimators = cross_validation(lR, parameters, X_usampled, y_usampled)
lRufit = lRuEstimators.fit(X_usampled, y_usampled)
print(lRufit.coef_)
print('Logistic Regression, Under Sampled: ', datetime.now() - start)
#
# Random Forest Under
rF = RandomForestClassifier()
parameters = [{'max_leaf_nodes': [8], 'max_depth': [6], 'min_samples_split': [8],
               'min_samples_leaf': [8]}]
start = datetime.now()
rfuEstimators = cross_validation(rF, parameters, X_usampled, y_usampled)
print(rfuEstimators.feature_importances_)
rFufit = rfuEstimators.fit(X_usampled, y_usampled)
print('Random Forest, Under Sampled: ', datetime.now() - start)
#
# Logistic Regression Over
parameters = [{}]
start = datetime.now()
pipeline = Pipeline([('smote_enn', sm),
                     ('clf_lr', lR)])
gs = GridSearchCV(pipeline, parameters, cv=10)
gs.fit(X_train, y_train)
model = gs.best_estimator_.fit(X_train, y_train)
print('Logistic Regression, Over Sampled: ', datetime.now() - start)

# # Random Forest Over
# parameters = [{'max_leaf_nodes': [8], 'max_depth': [8], 'min_samples_split': [8],
#                'min_samples_leaf': [8]}]
# start = datetime.now()
# rfoEstimators = cross_validation(rF, parameters, X_osampled, y_osampled)
# rFofit = rfoEstimators.fit(X_osampled, y_osampled)
# print('Random Forest, Over Sampled: ', datetime.now() - start)

# # Sequential Deep Learning Model
# dl_model = keras.models.Sequential([
#     keras.layers.InputLayer(input_shape=(30,)),
#     keras.layers.Dense(30, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# opt = SGD(learning_rate=0.001)
# dl_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# X_osampled, y_osampled = sm.fit_sample(X_train, y_train)
# df = pandas.concat((X_osampled, y_osampled), axis=1, join='inner')
# df = df.sample(frac=1).reset_index(drop=True)
# X_osampled = df.drop('Class', axis=1)
# y_osampled = df['Class']
#
# checkpoint = keras.callbacks.ModelCheckpoint('keras_model.h5', save_best_only=True)
# early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
#
# history = dl_model.fit(X_osampled, y_osampled, epochs=30, validation_split=0.2, shuffle=True,
#                        callbacks=[early_stopping, checkpoint])
# y_pred_o = dl_model.predict(X_test)
# print(list(y_pred_o))
# confusion_matrix(y_test, y_pred_o, labels=(1, 0))
# over_nn = accuracy_score(y_test, y_pred_o)

# history2 = dl_model.fit(X_usampled, y_usampled, epochs=30, validation_split=0.2, shuffle=True,
#                         callbacks=[early_stopping, checkpoint])
# y_pred_u = numpy.argmax(dl_model.predict(X_test), axis=-1)
# under_nn = accuracy_score(y_test, y_pred_u)

lRu_y = lRufit.predict(X_test)
print(confusion_matrix(y_test, lRu_y))
print(accuracy_score(y_test, lRu_y))
rFu_y = rFufit.predict(X_test)
print(accuracy_score(y_test, rFu_y))
lRo_y = model.predict(X_test)
print(accuracy_score(y_test, lRo_y))
# rFo_y = rFofit.predict(X_test)
# print(accuracy_score(y_test, rFo_y))
# print(over_nn)
# print(under_nn)