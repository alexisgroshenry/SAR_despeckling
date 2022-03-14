import os
import random
import shutil

if __name__ == "__main__" :
    # Directory of the data
    dir_name = "./GT_full"

    # Type of split
    by_pile = False

    # Folder containing the split
    folder_final = "./data"

    # Name of the piles
    piles = set()
    for file in os.listdir(dir_name) :
        pile_name = file.split("_")[0]
        piles.add(pile_name)
    piles = list(piles)

    if by_pile == True :
        # Split by piles
        # first three piles for train, fourth for val and last for test
        train_files = []
        validation_files = []
        test_files = []

        for file in os.listdir(dir_name) :
            pile_name = file.split('_')[0]
            idx = piles.index(pile_name)
            if idx < 3 :
                train_files.append(file+"\n")
            elif idx == 3 :
                validation_files.append(file+"\n")
            else :
                test_files.append(file+"\n")

        with open("train_split_GT.txt","w") as f :
            f.writelines(train_files)
        with open("validation_split_GT.txt","w") as f :
            f.writelines(validation_files)
        with open("test_split_GT.txt","w") as f :
            f.writelines(test_files)

    else :
        # Random split
        # Shuffle the files
        files =  os.listdir(dir_name)
        files = [f for f in files if f.endswith("npy")] # pairs together
        random.shuffle(files)

        train_count = int(0.7*len(files))
        val_count = int(0.8*len(files))

        train_files = []
        validation_files = []
        test_files = []


        for idx, file in enumerate(files) :
            if idx < train_count :
                train_files.append(file+"\n")
                train_files.append(file.replace("npy","png")+"\n")
            elif idx < val_count :
                validation_files.append(file+"\n")
                validation_files.append(file.replace("npy","png")+"\n")
            else :
                test_files.append(file+"\n")
                test_files.append(file.replace("npy","png")+"\n")

        with open("train_split_GT.txt","w") as f :
            f.writelines(train_files)
        with open("validation_split_GT.txt","w") as f :
            f.writelines(validation_files)
        with open("test_split_GT.txt","w") as f :
            f.writelines(test_files)

    # copy according to the split
    if not os.path.exists(folder_final) :
        os.mkdir(folder_final)

    print("Working on training set")
    folder = os.path.join(folder_final,"train")
    if not os.path.exists(folder) :
        os.mkdir(folder)
        
    with open("train_split_GT.txt", "r") as f :
        data = f.readlines()
        data = [f.replace("\n","") for f in data]

    for file in data :
        source = os.path.join(dir_name,file)
        destination = folder
        shutil.copy(source, destination)
        
    print("Working on validation set")
    folder = os.path.join(folder_final,"evaluation")
    if not os.path.exists(folder) :
        os.mkdir(folder)
        
    with open("validation_split_GT.txt", "r") as f :
        data = f.readlines()
        data = [f.replace("\n","") for f in data]

    for file in data :
        source = os.path.join(dir_name,file)
        destination = folder
        shutil.copy(source, destination)
        
    print("Working on testing set")
    folder = os.path.join(folder_final,"test")
    if not os.path.exists(folder) :
        os.mkdir(folder)
        
    with open("test_split_GT.txt", "r") as f :
        data = f.readlines()
        data = [f.replace("\n","") for f in data]

    for file in data :
        source = os.path.join(dir_name,file)
        destination = folder
        shutil.copy(source, destination)
