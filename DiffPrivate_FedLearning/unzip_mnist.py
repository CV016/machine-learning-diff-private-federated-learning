import gzip
import pickle
import os

def unzip_gzip(filename):
    # Unzip the .gz file and save the content as a new file
    with gzip.open(filename, 'rb') as f_in:
        data = pickle.load(f_in)
        
    # Create the output file name by removing the '.gz' extension
    uncompressed_filename = filename.replace('.gz', '')
    
    # Save the uncompressed data
    with open(uncompressed_filename, 'wb') as f_out:
        pickle.dump(data, f_out)

    print(f"Unzipped and saved: {uncompressed_filename}")

# Change to the MNIST_original directory
os.chdir('MNIST_original')

# List of gzipped files to unzip
gz_files = ['mnist_train_images.gz', 'mnist_train_labels.gz', 
            'mnist_test_images.gz', 'mnist_test_labels.gz']

# Unzip each file
for gz_file in gz_files:
    unzip_gzip(gz_file)
