import pickle

# Load the pickle file
with open('/Users/chaklader/Documents/Education/Udacity/Deep_Learning/Projects/4_face_generation/lectures/DCGAN/train_samples.pkl', 'rb') as f:
    data = pickle.load(f)

# Inspect the basic structure
print(type(data))           # Shows what type of object was pickled
print(dir(data))            # Shows available attributes/methods