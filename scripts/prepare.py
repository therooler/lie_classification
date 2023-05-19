import os

if __name__ == "__main__":

    print("Creating folder structure for data...")
    dim_list = [4, 8, 16, 32, 64, 128, 256]
    if not os.path.exists(f'./data/su4_raw'):
        os.makedirs(f'./data/su4_raw')
    if not os.path.exists(f'./data/su4_I_raw'):
        os.makedirs(f'./data/su4_I_raw')
    for dim in dim_list:
        if not os.path.exists(f'./data/su{dim}'):
            os.makedirs(f'./data/su{dim}')
        if not os.path.exists(f'./data/su{dim}_I'):
            os.makedirs(f'./data/su{dim}_I')
    if not os.path.exists(f'./figures'):
        os.makedirs(f'./figures')
    if not os.path.exists(f'./tests'):
        os.makedirs(f'./tests')
    print("Done!")
