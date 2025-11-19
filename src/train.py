from data_prep import prepare_data
from model import train_leaf_detector, train_species_classifier

if __name__ == "__main__":
    data = prepare_data()

    print("Training leaf detector...")
    leaf_model, leaf_history = train_leaf_detector(data['leaf_detector'])
    print("Leaf detector trained and saved.")

    print("Training species classifier...")
    species_model, species_history = train_species_classifier(data['species_classifier'])
    print("Species classifier trained and saved.")
