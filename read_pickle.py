import pickle
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def read_pickle_file(file_path, output_path, output_format='JSON'):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
        if output_format == 'json':
            with open(output_path, 'w') as json_file:
                json.dump(data, json_file, indent=4, cls=NumpyEncoder)
            print(f"Data has been exported to {output_path} as JSON.")
        elif output_format == 'txt':
            with open(output_path, 'w') as txt_file:
                txt_file.write(str(data))
            print(f"Data has been exported to {output_path} as TXT.")
        else:
            print("Unsupported output format. Please use 'json' or 'txt'.")

if __name__ == "__main__":
    pickle_file_path = "./experiments/exp_03182025_10:05:21/pose_point_map.pickle"  # Change this to your pickle file path
    output_file_path = "./experiments/exp_03182025_10:05:21/pose_point_map.json"  # Change this to your desired output file path
    read_pickle_file(pickle_file_path, output_file_path, output_format='json')