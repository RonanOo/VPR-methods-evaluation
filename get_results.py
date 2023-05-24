import os
import pickle

def load_and_concatenate(directory):
    result = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'rb') as f:
                dict_data = pickle.load(f)
                
                for key, value in dict_data.items():
                    if key in result:
                        result[key] += value
                    else:
                        result[key] = value
    
    return result

directory = 'results'  # The directory containing your pickle files
final_dict = load_and_concatenate(directory)

# Print the final dictionary
for key, value in final_dict.items():
    print(f'{key}: {value}')
