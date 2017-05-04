import pandas as pd
import urllib
import ntpath
import os

DELIMITER = ','
MAX_INDEX = '1000000'


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def load_datasets(urls, output_directory):
    for url in urls:
        print("downloading " + url)
        path = path_leaf(url)
        urllib.urlretrieve(url, filename=os.path.join(base_directory, path))

def isPositiveFloat(value):
  try:
    floatedValue = float(value)
    if floatedValue <= 0:
        return False
    return True
  except ValueError:
    return False

from collections import OrderedDict

def convert_data(input_paths, output_prefix, num_of_columns, columns_to_convert):
    key_store_list = {}
    output_file_list = {}
    for column in columns_to_convert:
        index_file = open(output_prefix + '_' + str(column), 'w')
        output_file_list[column] = index_file
        key_store_list[column] = OrderedDict()

    for input_path in input_paths:
        line_cnt = 0
        with open(input_path, "r") as f:
            for line in f:
                if line_cnt != 0:
                    entries = line.strip().split(',')
                    if len(entries) != num_of_columns:
                        continue

                    for column in columns_to_convert:
                        entry = entries[int(column)].split(' ')
                        key = ''
                        for value in entry:
                            key += value
                        key_store_list[column][key] = True

                line_cnt += 1

    for column in columns_to_convert:
        key_cnt = 0
        for entry in key_store_list[column].keys():
            output_file_list[column].write(entry.strip() + ',' + str(key_cnt) + "\n")
            key_cnt += 1

    for column in columns_to_convert:
        output_file_list[column].close()


# read multiple files and merge into one file after pre-processing.
def run_preprocessing(input_paths, output_path, error_path, index_prefix, num_of_columns, columns_to_convert):

    key_store_list = {}
    for column in columns_to_convert:
        with open(index_prefix + '_' + str(column), "r") as f:
            dictionary = {}
            for line in f:
                keyAndValue = line.split(DELIMITER)
                dictionary[keyAndValue[0]] = keyAndValue[1]
            key_store_list[column] = dictionary

    output_file = open(output_path, 'w')
    error_file = open(error_path, 'w')

    is_first_file = True
    for input_path in input_paths:
        line_cnt = 0
        with open(input_path, "r") as f:
            for line in f:
                if line_cnt == 0:
                    if is_first_file:
                        output_file.write(line.strip() + "\n")
                        error_file.write(line.strip() + "\n")
                        is_first_file = False
                else:
                    entries = line.strip().split(DELIMITER)

                    if len(entries) != num_of_columns or entries[12] == 'none' or entries[13] == 'none' \
                            or entries[12] == '0;0' or entries[13] == '0;0':
                        error_file.write(line.strip() + "\n")
                        continue

                    values = entries[12].split(';')
                    if len(values) != 2 or isPositiveFloat(values[0]) != True or isPositiveFloat(values[1]) != True:
                        error_file.write(line.strip() + "\n")
                        continue

                    values = entries[13].split(';')
                    if len(values) != 2 or isPositiveFloat(values[0]) != True or isPositiveFloat(values[1]) != True:
                        error_file.write(line.strip() + "\n")
                        continue

                    final_output = ''
                    for index, item in enumerate(entries):
                        if index in columns_to_convert:
                            entry = entries[index].split(' ')
                            key = ''
                            for value in entry:
                                key += value
                            value = key_store_list[index].get(key, MAX_INDEX)
                            final_output += value.strip()
                        else:
                            final_output += item

                        final_output += DELIMITER

                    final_output = final_output.strip()
                    final_output = final_output[0:len(final_output) - 1]
                    output_file.write(final_output + "\n")

                line_cnt += 1

    output_file.close()
    error_file.close()

def run_training(input_path, model_path, training_stats_path):

    na_dictionary = {'delivery_coordinates': ['none'], 'scheduled_coordinates': ['none']}
    df = pd.read_csv(input_path, na_values=na_dictionary)
    df['scheduled_coordinates'] = df['scheduled_coordinates'].replace('none', '0;0')
    df = df.dropna(subset=['delivery_coordinates'])
    df = df.dropna(subset=['scheduled_coordinates'])

    customer_name = df['customer_name']
    customer_phone = df['customer_phone']
    customer_address = df['customer_address'].astype('category')
    supplier_name = df['supplier_name'].astype('category')
    schedule_channel = df['schedule_channel'].astype('category')

    del_coords = df['delivery_coordinates']
    del_coords_x = [i.split(';')[0] for i in del_coords]
    del_coords_y = [i.split(';')[1] for i in del_coords]
    sch_coords = df['scheduled_coordinates']
    sch_coords_x = [i.split(';')[0] for i in sch_coords]
    sch_coords_y = [i.split(';')[1] for i in sch_coords]

    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0,1))
    #
    # scaled_del_coords_x = scaler.fit_transform(del_coords_x)
    # scaled_del_coords_y = scaler.fit_transform(del_coords_y)
    # scaled_sch_coords_x = scaler.fit_transform(sch_coords_x)
    # scaled_sch_coords_y = scaler.fit_transform(sch_coords_y)
    #
    # merged_coords = pd.DataFrame(
    #     list(map(list, zip(customer_name,
    #                        customer_phone,
    #                        customer_address,
    #                        supplier_name,
    #                        schedule_channel,
    #                        scaled_sch_coords_x,
    #                        scaled_sch_coords_y,
    #                        scaled_del_coords_x,
    #                        scaled_del_coords_y
    #                        ))))

    merged_coords = pd.DataFrame(
        list(map(list, zip(customer_name,
                           customer_phone,
                           customer_address,
                           supplier_name,
                           schedule_channel,
                           sch_coords_x,
                           sch_coords_y,
                           del_coords_x,
                           del_coords_y
                           ))))

    import random
    X = merged_coords[[2, 3, 4, 5, 6]]
    Y = merged_coords[[7, 8]]

    rows = random.sample(merged_coords.index, int(len(merged_coords) * .80))
    x_train, y_train = X.ix[rows], Y.ix[rows]
    x_test, y_test = X.drop(rows), Y.drop(rows)

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    params = {'n_estimators': 500, 'max_depth': 6,
              'learning_rate': 0.1, 'loss': 'huber', 'alpha': 0.95}
    clf = MultiOutputRegressor(GradientBoostingRegressor(**params)).fit(x_train, y_train)

    mse = mean_squared_error(y_test, clf.predict(x_test))
    mae = mean_absolute_error(y_test, clf.predict(x_test))
    r2 = r2_score(y_test, clf.predict(x_test))

    print("MSE: %.8f" % mse)
    print("MAE: %.8f" % mae)
    print("R2: %.8f" % r2)

    from sklearn.externals import joblib
    joblib.dump(clf, model_path)

    with open(training_stats_path, "w") as f:
        f.write("MSE: %.8f" % mse + "\n")
        f.write("MAE: %.8f" % mae + "\n")
        f.write("R2: %.8f" % r2 + "\n")
        f.write("max_delivered_coords_x: " + max(del_coords_x) + "\n")
        f.write("min_delivered_coords_x: " + min(del_coords_x) + "\n")
        f.write("max_delivered_coords_y: " + max(del_coords_y) + "\n")
        f.write("min_delivered_coords_y: " + min(del_coords_y) + "\n")
        f.write("max_scheduled_coords_x: " + max(sch_coords_x) + "\n")
        f.write("min_scheduled_coords_x: " + min(sch_coords_x) + "\n")
        f.write("max_scheduled_coords_y: " + max(sch_coords_y) + "\n")
        f.write("min_scheduled_coords_y: " + min(sch_coords_y) + "\n")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--input', help='Local Base Directory', required=True)
    parser.add_argument('-skip', '--skip', help='skip downloading data-set', required=True)
    args = parser.parse_args()

    print("fetchr_train's Main() is invoked")

    base_directory = args.input

    # step1: downloading data-set
    ##########################################################################################
    urls = ['https://s3-eu-west-1.amazonaws.com/fetchr-datascience/anon_dataset_01_2017.csv',
            'https://s3-eu-west-1.amazonaws.com/fetchr-datascience/anon_dataset_12_2016.csv',
            'https://s3-eu-west-1.amazonaws.com/fetchr-datascience/anon_dataset_11_2016.csv',
            'https://s3-eu-west-1.amazonaws.com/fetchr-datascience/anon_dataset_10_2016.csv']

    if args.skip != 'true':
        print('downloading dataset...')
        load_datasets(urls, base_directory)
        print('downloading is done.')
    ##########################################################################################

    raw_input_paths = []
    for url in urls:
        path = path_leaf(url)
        raw_input_paths.append(os.path.join(base_directory, path))
        

    # step2: preprocessing data-set
    ##########################################################################################
    index_prefix = os.path.join(base_directory, 'index')
    num_of_columns = 14
    columns_to_convert = [4, 9, 10]

    print('preprocessing dataset...')
    convert_data(raw_input_paths, index_prefix, num_of_columns, columns_to_convert)

    preprocessing_output_path = os.path.join(base_directory, 'preprocessed_data')
    preprocessing_error_path = os.path.join(base_directory, 'preprocessed_err_data')

    run_preprocessing(raw_input_paths,
                      preprocessing_output_path,
                      preprocessing_error_path,
                      index_prefix,
                      num_of_columns,
                      columns_to_convert)
    print('preprocessing is done.')
    ##########################################################################################

    # step3: training data-set
    ##########################################################################################
    print('training dataset...')
    model_path = os.path.join(base_directory, 'model.pkl')
    training_stats_path = os.path.join(base_directory, 'training_stats')

    run_training(preprocessing_output_path,
                 model_path,
                 training_stats_path)
    print('training is done.')
    ##########################################################################################

else:
    print("fetchr_train is imported")