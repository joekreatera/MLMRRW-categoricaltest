#modules!
import os
import tarfile
from six.moves import urllib
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy
# another trick to get total uri.
# scriptpath = os.path.realpath(__file__)

download_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
path = "datasets/housing"
housing_url_base = download_url + path + "/housing.tgz";


def saveFile(dataframe, uri , idx=False):
    dataframe.to_csv(path_or_buf=uri, index=idx)


def getFromOpenML(database,version="active",ospath="datasets/", download = False, save=True):

    instances,instances_labels = 0,0

    if download:
        
        if not os.path.isdir(ospath):
            os.makedirs(ospath)
            
        print(f"Downloading {database}")
        X,Y = fetch_openml(database, version=version, return_X_y=True)
        print(f"Downloaded {database}")
        
        instances = pd.DataFrame.from_records(X)
        
        print(type(Y))
        print((Y.shape))
        if type(Y) ==   numpy.ndarray:
            if( len(Y.shape) == 0 ):
                Y = pd.Series(Y)
            else:
                Y = pd.DataFrame(Y)
            
        if( type(Y) == pd.Series  ):
            instances_labels = Y.to_frame()
        else:
            instances_labels = Y
            instances_labels.replace({'FALSE':0,'TRUE':1} , inplace=True)
        
        inst_cols = instances.columns

        newNames = []
        for i in range(0,len(inst_cols) ):
            name= inst_cols[i]

            newNames.append( "".join([ "col_",str(i) ]) )


        instances.columns = newNames

        labels_cols = instances_labels.columns
        
        newNames = []
        for i in range(0, len(labels_cols) ):
            name= labels_cols[i]
            # print(name)
            newNames.append( "".join(["label_", str(i) ]))

        instances_labels.columns = newNames
    else:
        to_load = "".join([ospath,database,".csv"])
        return pd.read_csv( to_load)

    full_database = pd.concat([instances, instances_labels], axis=1)
    full_uri_to_save = ""
    full_uri_to_save = full_uri_to_save.join([ospath,database,".csv"])
    if save:
        full_database.to_csv(path_or_buf=full_uri_to_save, index=False)

    return full_database


def getFullURI(fl):
    return  os.path.join(os.getcwd() , fl )

def getData( housing_url = housing_url_base, housing_path=path):

    print("Downloading from " + housing_url_base)

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    tgz_path= os.path.join( os.getcwd() , tgz_path)

    returned = urllib.request.urlretrieve(housing_url,tgz_path)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
