import pandas as pd
import math
import pickle
import sys
import re

class dist_encode():
    def __init__(self, encode_type = None):
        """
        dateNames: list of date columns
        only binary encode now
        """
        encode_types = ["binary"]
        if encode_type not in encode_types:
            exit("invalid encode type")

    def gen_col_map(self, fea, num_digits):
        '''
        fea: feature name
        num_digits: number of digits used to encode this feature
        return: like feature B used 4 digits, so encoded B columns -> [B_0, B_1, B_2, B_3]
        '''
        col_map = []
        for i in range(num_digits):
            col_map.append(fea + '_' + str(i))
        return col_map

    def get_cate_data(self, df):
        """
        df: input data frame
        return: headers for obj data
        """
        obj_headers = df.select_dtypes(include=['object']).columns.values.tolist()
        i = 0
        while i < len(obj_headers):
            try:
                df[obj_headers[i]] = pd.to_datetime(df[obj_headers[i]])
                print("%s is datetime" % obj_headers[i])
                del obj_headers[i]
            except ValueError:
                print("%s not datetime" % obj_headers[i])
                i += 1
                continue
        print(obj_headers)
        return obj_headers

    def gen_map(self, df, headers):
        """
        df: data frame
        headers: obj data column names
        return: maps from class to code / maps from column name to sub column names
        """
        maps = {}
        col_maps = {}
        for fea in headers:
            encode_map = dict()
            cla = df[fea].value_counts()
            num_digits = int(math.ceil(math.log(len(cla), 2)))
            col_maps[fea] = self.gen_col_map(fea, num_digits)
            print("new columns for {}: ".format(fea))
            print(col_maps[fea])
            for i in range(len(cla)):
                encode_map[cla.keys()[i]] = bin(i)[2:].zfill(num_digits)
            print("map for %s is %s" % (fea, encode_map))
            maps[fea] = encode_map
        try:
            with open("./maps.txt", "wb") as fp:
                pickle.dump(maps, fp)
        except:
            print("save map fail!")
        try:
            with open("./col_maps.txt", "wb") as fp:
                pickle.dump(col_maps, fp)
        except:
            print("save map fail!")
        return maps, col_maps

    def inverse_map(self, maps):
        """
        maps: dict from class to code
        return: dict from code to class
        """
        inverse_maps = {}
        for fea in maps.keys():
            map = maps[fea]
            new_map = {v: k for k, v in map.items()}
            inverse_maps[fea] = new_map
        try:
            with open("./inverse_maps.txt", "wb") as fp:
                pickle.dump(inverse_maps, fp)
        except:
            print("save inverse map fail!")
        return inverse_maps

    def fit(self, df_train, encode_type = "binary"):
        """
        data_dir: dir for input train data
        encode_type: binary, haskell...
        return: map from class to bin code, map from bin code to class, map of columns to sub columns
        """
        #df_train = pd.read_csv(data_dir)
        headers_train = df_train.columns.values.tolist()
        obj_headers = self.get_cate_data(df_train)
        maps, col_maps = self.gen_map(df_train, obj_headers)
        inverse_maps = self.inverse_map(maps)
        return maps, inverse_maps, col_maps

    def transform(self, data, maps = None, maps_dir = None, col_maps = None):
        """
        data: dataframe to transform
        maps: dict mapping get from fit()
        maps_dir(optional): dir for maps
        return: encoded obj data
        """
        if maps:
            maps = maps
        elif maps_dir:
            try:
                with open(maps_dir, "rb") as fp:
                    maps = pickle.load(fp)
            except FileNotFoundError as error:
                print(error)
        else:
            exit("Missing maps!")

        #data = pd.read_csv(data)
        obj_headers = self.get_cate_data(data)
        # assert obj_headers == maps.keys()
        for fea in obj_headers:
            col_names = col_maps[fea]
            map = maps[fea]
            for cls in map.keys():
                data[fea] = data[fea].replace(cls, map[cls])
            for i in range(len(col_names)):
                data.loc[:, col_names[i]] = data[fea].map(lambda x: x[i])
            data = data.drop(columns = fea)
        return data

    def inverse_transform(self, data, inverse_maps = None, inv_maps_dir = None, col_maps = None):
        """
        data: dir to encoded data (sub columns and shuffled)
        maps: dict mapping
        maps_dir(optional): dir to inverse map
        return: decoded obj dataframe
        """
        if inverse_maps:
            inverse_maps = inverse_maps
        elif inv_maps_dir:
            try:
                with open(inv_maps_dir, "rb") as fp:
                    inverse_maps = pickle.load(fp)
            except FileNotFoundError as error:
                print(error)
        else:
            exit("Missing inverse maps!")
        # data = pd.read_csv(data)
        sub_obj_headers = self.get_cate_data(data)
        for fea in col_maps.keys():
            sub_cols = []
            for sub_col in sub_obj_headers:
                if re.match(('^' + fea + '_'), sub_col):
                    sub_cols.append(sub_col)
            sub_cols.sort()
            print("Found these columns related to feature {}:".format(fea))
            print(sub_cols)
            for i in range(len(data[sub_col])):
                data.loc[i, fea] = ''.join([data[x][i] for x in sub_cols])
            data = data.drop(columns = sub_cols)
        obj_headers = col_maps.keys()
        for fea in obj_headers:
            inverse_map = inverse_maps[fea]
            for cls in inverse_map.keys():
                data[fea] = data[fea].replace(cls, inverse_map[cls])
        return data


if __name__ == "__main__":
    train_dir = "./train.csv"
    # test_dir = "./encode_test.csv"
    encoder = dist_encode("binary")
    maps, inverse_maps = encoder.fit(train_dir, "binary")
    obj_data = encoder.transform(train_dir, maps_dir = "./maps.txt")
    inv_data = encoder.inverse_transform(obj_data, inv_maps_dir = "./inverse_maps.txt")
