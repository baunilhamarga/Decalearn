import numpy as np

def print_statistics(df, classification=False, check_content=False):
    X_train, y_train, X_test, y_test = df['X_train'], df['y_train'], df['X_test'], df['y_test'] 

    # Rebuild dataFrame
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test), axis=0)

    # Calculate the number of samples (rows) and features (columns)
    num_samples, num_features = X.shape

    if classification:
        # 'y' column in the DataFrame that represents classes (target feature)
        num_classes = len(np.unique(y))  # Count the number of unique classes

        # Calculate the minimum and maximum class sizes
        unique_classes, class_counts = np.unique(y, return_counts=True)

        min_class_size = class_counts.min()
        max_class_size = class_counts.max()

    # Calculate train and test sizes and relative sizes
    train_size = len(y_train)
    train_size_percentage = 100*train_size/num_samples
    test_size = len(y_test)
    test_size_percentage = 100*test_size/num_samples

    # Check the saved arrays shape and content
    if check_content:
        print(f"X_train.shape:{X_train.shape}")
        print(X_train)
        print(f"y_train.shape:{y_train.shape}")
        print(y_train)
        print(f"X_test.shape:{X_test.shape}")
        print(X_test)
        print(f"y_test.shape:{y_test.shape}")
        print(y_test)

    # Print the statistics
    print("#Samples:", num_samples)
    print("#Features:", num_features)
    if classification:
        print("#Classes:", num_classes)
        print("#Min class size:", min_class_size)
        print("#Max class size:", max_class_size)
    print(f"Train size: {train_size} ({train_size_percentage:.0f}%)")
    print(f"Test size: {test_size} ({test_size_percentage:.0f}%)")

if __name__ == '__main__':
    data = np.load('/home/baunilha/Repositories/Decalearn/Datasets/Multimodal Human Action/data/UTD-MHAD2_1s.npz')
    print_statistics(data, check_content=True, classification=True)