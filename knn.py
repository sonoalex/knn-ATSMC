import pandas as pd
from normalizer import Normalizer
import csv
from sklearn.model_selection import train_test_split


def distance(track1, track2):
    '''
    Computes euclidean distance ffrom two tracks at n points
    Input:
        - track1: list
        - track2: list
    Output: 
        distance: float
    '''
    squared_difference = 0

    for i in range(len(track1)):
        squared_difference += (float(track1[i]) - float(track2[i])) ** 2
    final_distance = squared_difference ** 0.5
    return final_distance

def classify(unknown, dataset, labels, k):
    distances = []
    #Looping through all points in the dataset
    for filename in dataset:
        track = dataset[filename]
        distance_to_point = distance(track, unknown)
        #Adding the distance and point associated with that distance
        distances.append([distance_to_point, filename])
    distances.sort()
    #Taking only the k closest points
    neighbors = distances[:k]

    results = {'blues' : 0, 'hiphop' : 0, 'classical' : 0, 'country' : 0, 'disco' : 0, 'jazz': 0, 'metal': 0, 'pop': 0, 'reggae': 0, 'rock': 0}

    for neighbor in neighbors:
        filename = neighbor[1]
        results[labels[filename]]+= 1
    
    max_value = max(results, key=results.get)

    return max_value

def normalize_data_and_generate_csv(filename):
    new_filename = 'normalized_' + filename
    
    df = pd.read_csv(filename, names=['zcr', 'loud', 'sct', 'energy', 'rms', 'filename', 'label'])
    
    pd.set_option('display.max_rows', df.shape[0]+1)
    df = df.sort_values('filename')
    normalizer = Normalizer()
    zcr_normalized = normalizer.min_max_normalize(df['zcr'].tolist())
    loud_normalized = normalizer.min_max_normalize(df['loud'].tolist())
    sct_normalized = normalizer.min_max_normalize(df['sct'].tolist())
    energy_normalized = normalizer.min_max_normalize(df['energy'].tolist())
    rms_normalized = normalizer.min_max_normalize(df['rms'].tolist())

    df['zcr'] = pd.DataFrame(zcr_normalized)
    df['loud'] = pd.DataFrame(loud_normalized)
    df['sct'] = pd.DataFrame(sct_normalized)
    df['energy'] = pd.DataFrame(energy_normalized)
    df['rms'] = pd.DataFrame(rms_normalized)
    df_labels = df[['filename','label']]
    df.to_csv(new_filename, encoding='utf-8', index=False, header=False)
    df_labels.to_csv('track_labels.csv', encoding='utf-8', index=False, header=False)


def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
  num_correct = 0.0
  for title in validation_set:
    guess = classify(validation_set[title], training_set, training_labels, k)
    if guess == validation_labels[title]:
      num_correct+=1
  return num_correct/len(validation_set)
  

#print(find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, 3))
#normalize_data_and_generate_csv('data_genres_features.csv')
with open('normalized_data_genres_features.csv', newline='') as f:
    reader = csv.reader(f)
    X = list(reader)
with open('track_labels.csv', newline='') as f:
    reader = csv.reader(f)
    y = list(reader)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = {}
y_train = {}
X_test = {}
y_test = {}
for x in X_train_raw:
    X_train.update({x[5]:x[:5]})
for y in y_train_raw:
    y_train.update({y[0]:y[1]})

for x in X_test_raw:
    X_test.update({x[5]:x[:5]})
for y in y_test_raw:
    y_test.update({y[0]:y[1]})


print(find_validation_accuracy(X_train, y_train, X_test, y_test, 8))

'''for k in range(200):
    print(find_validation_accuracy(X_train, y_train, X_test, y_test, k))'''



