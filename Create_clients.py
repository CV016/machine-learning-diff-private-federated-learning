import pickle
import numpy as np
import os

def create_clients(num, dir):
    '''
    This function creates clients that hold non-IID MNIST data according to the experiments in 
    https://research.google.com/pubs/pub44822.html. It generates indices that point to data, 
    creating a non-IID distribution for each client.
    
    :param num: Number of clients
    :param dir: Directory where to store the client data
    :return: None
    '''

    num_examples = 50000
    num_classes = 10
    file_path = os.path.join(dir, f'{num}_clients.pkl')

    if os.path.exists(file_path):
        print(f'Client exists at: {file_path}')
        return

    if not os.path.exists(dir):
        os.makedirs(dir)

    buckets = []
    for k in range(num_classes):
        temp = []
        for j in range(num // 100):  # Use integer division
            temp = np.concatenate((temp, k * num_examples // 10 + np.random.permutation(num_examples // 10)))
        buckets = np.concatenate((buckets, temp))

    shards = 2 * num
    perm = np.random.permutation(shards)
    z = []
    ind_list = np.split(buckets, shards)

    for j in range(0, shards, 2):
        z.append(np.concatenate((ind_list[int(perm[j])], ind_list[int(perm[j + 1])])))
        perm_2 = np.random.permutation(len(z[-1]))
        z[-1] = z[-1][perm_2]

    try:
        with open(file_path, "wb") as filehandler:
            pickle.dump(z, filehandler)
        print(f'Client created at: {file_path}')
    except Exception as e:
        print(f'Error saving client data: {e}')

if __name__ == '__main__':
    List_of_clients = [100, 200, 500, 1000, 2000, 5000, 10000]
    for j in List_of_clients:
        create_clients(j, os.path.join(os.getcwd(), 'DATA', 'clients'))
