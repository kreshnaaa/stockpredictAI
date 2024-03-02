from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras


iris = load_iris()
print(iris)
X = iris.data
y = iris.target

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the target variable
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Create the neural network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])

# Train the neural network
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=10, validation_split=0.2)

# Evaluate the neural network
test_loss, test_acc = model.evaluate(X, y)
print('Test accuracy:', test_acc)