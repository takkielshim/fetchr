import pandas as pd

DELIMITER = ','

def run_prediction(model_path, input_path, columns_to_convert, output_path):

    key_store_list = {}
    for column in columns_to_convert:
        with open(index_prefix + '_' + str(column), "r") as f:
            dictionary = {}
            for line in f:
                keyAndValue = line.split(DELIMITER)
                dictionary[keyAndValue[0]] = keyAndValue[1]
            key_store_list[column] = dictionary

    df = pd.read_csv(input_path)

    customer_name = df['customer_name']
    customer_phone = df['customer_phone']
    customer_address = df['customer_address'].astype('category')
    supplier_name = df['supplier_name'].astype('category')
    schedule_channel = df['schedule_channel'].astype('category')

    sch_coords = df['scheduled_coordinates']
    sch_coords_x = [i.split(';')[0] for i in sch_coords]
    sch_coords_y = [i.split(';')[1] for i in sch_coords]

    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
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
    #                        scaled_sch_coords_y
    #                        ))))

    merged_coords = pd.DataFrame(
        list(map(list, zip(customer_name,
                           customer_phone,
                           customer_address,
                           supplier_name,
                           schedule_channel,
                           sch_coords_x,
                           sch_coords_y
                           ))))

    X = merged_coords[[2, 3, 4, 5, 6]]

    from sklearn.externals import joblib
    clf = joblib.load(model_path)
    result = clf.predict(X)

    with open(output_path, "w") as f:
        for index, (value1, value2, value3) in enumerate(zip(merged_coords[0], merged_coords[1], result)):
            f.write(str(value1) + '_' + str(value2) + ':' + str(value3) + "\n")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--input', help='Local Base Directory', required=True)
    parser.add_argument('-data', '--data', help='Test(Predict) File Name', required=True)
    args = parser.parse_args()

    print("fetchr_predict's Main() is invoked")
    import os

    base_directory = args.input
    raw_input_paths = [os.path.join(base_directory, args.data)]

    index_prefix = os.path.join(base_directory, 'index')
    num_of_columns = 14
    columns_to_convert = [4, 9, 10]

    preprocessing_output_path = os.path.join(base_directory, 'preprocessed_predict_data')
    preprocessing_error_path = os.path.join(base_directory, 'preprocessed_predict_err_data')

    model_path = os.path.join(base_directory, 'model.pkl')
    predict_output_path = os.path.join(base_directory, 'prediction_result')

    from fetchr_train import run_preprocessing

    run_preprocessing(raw_input_paths,
                      preprocessing_output_path,
                      preprocessing_error_path,
                      index_prefix,
                      num_of_columns,
                      columns_to_convert)

    run_prediction(model_path,
                   preprocessing_output_path,
                   columns_to_convert,
                   predict_output_path)

else:
    print("fetchr_predict is imported")