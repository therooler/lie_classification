import os

if __name__ == "__main__":

    print("Creating folder structure for data...")
    dim_list = [4, 8, 16, 32, 64, 128, 256]
    for open_or_closed in ["open", "closed"]:
        prepend = f'./data/{open_or_closed}'
        if not os.path.exists(prepend + '/su4_raw'):
            os.makedirs(prepend + '/su4_raw')
        if not os.path.exists(prepend+'/su4_I_raw'):
            os.makedirs(prepend + '/su4_I_raw')
        for dim in dim_list:
            if not os.path.exists(prepend + f'/su{dim}'):
                os.makedirs(prepend + f'/su{dim}')
            if not os.path.exists(prepend + f'/su{dim}_I'):
                os.makedirs(prepend + f'/su{dim}_I')
        if not os.path.exists(f'./figures/{open_or_closed}'):
            os.makedirs(f'./figures/{open_or_closed}')
        if not os.path.exists(f'./tests'):
            os.makedirs(f'./tests')
    print("Done!")
