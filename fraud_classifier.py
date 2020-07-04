import pandas
from datetime import datetime
import numpy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from plots import plot_3d

data = pandas.read_csv('creditcard.csv')

# Imbalanced dataset so will need to resample
print(data['Class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data.Class, test_size=0.25,
                                                    random_state=42)
# Under Sampling to deal with imbalance
rus = RandomUnderSampler(random_state=42)
X_usampled, y_usampled = rus.fit_resample(X_train, y_train)

# Over Sampling to deal with imbalance
ros = RandomOverSampler(random_state=42)
X_osampled, y_osampled = ros.fit_resample(X_train, y_train)

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=X_osampled['Time'], y=X_osampled['Amount'], mode='markers'))
# fig.show()

# Scaling X variables to between 1 and 0 for PCA
scaler = MinMaxScaler()
X_usampled = pandas.DataFrame(scaler.fit_transform(X_usampled))
X_osampled = pandas.DataFrame(scaler.fit_transform(X_osampled))

# PCA
pca = PCA(n_components=3)
Xu_train_pca = pandas.DataFrame(pca.fit_transform(X_usampled))
Xo_train_pca = pandas.DataFrame(pca.fit_transform(X_osampled))


# Cross Val function to do GridSearchCV
def cross_validation(model, params, X_train, Y_train):
    f1_scorer = make_scorer(f1_score, pos_label=1)
    gs = GridSearchCV(model, params, cv=10, scoring=f1_scorer)
    gs.fit(X_train, Y_train)
    print('F1 Score: ')
    print(gs.best_score_)
    print(gs.best_estimator_)
    return gs.best_estimator_


# # Logistic Regression Under
# lR = LogisticRegression()
# parameters = [{}]
# start = datetime.now()
# lRuEstimators = cross_validation(lR, parameters, Xu_train_pca, y_usampled)
# lRufit = lRuEstimators.fit(Xu_train_pca, y_usampled)
# print(lRufit.coef_)
# print('Logistic Regression, Under Sampled: ', datetime.now() - start)
# #
# # Random Forest Under
# rF = RandomForestClassifier()
# parameters = [{'max_leaf_nodes': [8], 'max_depth': [6], 'min_samples_split': [8],
#                'min_samples_leaf': [8]}]
# start = datetime.now()
# rfuEstimators = cross_validation(rF, parameters, Xu_train_pca, y_usampled)
# print(rfuEstimators.feature_importances_)
# rFufit = rfuEstimators.fit(Xu_train_pca, y_usampled)
# print('Random Forest, Under Sampled: ', datetime.now() - start)
# #
# # Logistic Regression Over
# parameters = [{}]
# start = datetime.now()
# lRoEstimators = cross_validation(lR, parameters, Xo_train_pca, y_osampled)
# lRofit = lRoEstimators.fit(Xo_train_pca, y_osampled)
# print('Logistic Regression, Over Sampled: ', datetime.now() - start)
#
# # Random Forest Over
# parameters = [{'max_leaf_nodes': [8], 'max_depth': [8], 'min_samples_split': [8],
#                'min_samples_leaf': [8]}]
# start = datetime.now()
# rfoEstimators = cross_validation(rF, parameters, Xo_train_pca, y_osampled)
# rFofit = rfoEstimators.fit(Xo_train_pca, y_osampled)
# print('Random Forest, Over Sampled: ', datetime.now() - start)
#
# # Sequential Deep Learning Model
# model = keras.models.Sequential([
#     keras.layers.InputLayer(input_shape=(3,)),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(30, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
#
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#
# checkpoint = keras.callbacks.ModelCheckpoint('keras_model.h5')
# early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
#
# X_test = pandas.DataFrame(scaler.fit_transform(X_test))
# X_test = pandas.DataFrame(pca.fit_transform(X_test))
#
# history = model.fit(Xo_train_pca, y_osampled, epochs=30, validation_split=0.25, callbacks=[checkpoint, early_stopping])
# y_pred_o = numpy.argmax(model.predict(X_test), axis=-1)
# over_nn = accuracy_score(y_test, y_pred_o)
# history2 = model.fit(Xu_train_pca, y_usampled, epochs=30, validation_split=0.1, callbacks=[checkpoint, early_stopping])
# y_pred_u = numpy.argmax(model.predict(X_test), axis=-1)
# under_nn = accuracy_score(y_test, y_pred_u)
#
# lRu_y = lRufit.predict(X_test)
# print(accuracy_score(y_test, lRu_y))
# rFu_y = rFufit.predict(X_test)
# print(accuracy_score(y_test, rFu_y))
# lRo_y = lRofit.predict(X_test)
# print(accuracy_score(y_test, lRo_y))
# rFo_y = rFofit.predict(X_test)
# print(accuracy_score(y_test, rFo_y))
# print(over_nn)
# print(under_nn)


Xu_train_pca.columns = ['pca_one', 'pca_two', 'pca_three']

plot_3d(Xu_train_pca, y_usampled)