import os

def get_historical_images(img_path,):
    ages = []
    genders = []
    races =  []
    files = []
    for filename in os.listdir(os.path.join(img_path)):
        #avoid macos prob
        if('ds' in filename):
            continue
        path = os.path.join(img_path,filename)
        files.append(str(path))
        try:
            name = filename.split("_")
            age = int(name[0])
            gender = int(name[1])
            race = int(name[2])
            ages.append(age)
            genders.append(gender)
            races.append(race)
        except:
            print(filename)


    return ages,genders,races,files


ages,genders,races,files = get_utk_files('./UTKFace/')
