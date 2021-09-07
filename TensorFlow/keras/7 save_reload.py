from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
np.random.seed(1337)  # for reproducibility


# a fitting curves model
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]
model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)

# save mdoel
print('test before save:', model.predict(X_test[0:2]))
# model 的保存格式为 HDF5, 因此需要 pip 安装 h5py 包
model.save('my_model.h5')

del model  # delete the existing model

# load model
model = load_model('my_model.h5')
print('test after load: ', model.predict(X_test[0:2]))

"""
# save and load weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')

# save and load fresh network without trained weights
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
"""
