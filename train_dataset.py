import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

processed_data = []
processed_labels = []

for i in range(len(data)):

    sample = data[i]

    # Ensure correct feature length
    if len(sample) != 42:
        continue

    x_ = sample[0::2]
    y_ = sample[1::2]

    min_x = min(x_)
    min_y = min(y_)

    normalized_sample = []

    for j in range(len(x_)):
        normalized_sample.append(x_[j] - min_x)
        normalized_sample.append(y_[j] - min_y)

    processed_data.append(normalized_sample)
    processed_labels.append(labels[i])

data = np.array(processed_data)
labels = np.array(processed_labels)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels
)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)

model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)

score = accuracy_score(y_test, y_pred)

print(f'Accuracy: {score*100:.2f}%')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model trained and saved successfully.")