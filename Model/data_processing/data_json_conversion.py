import pandas as pd
import os
import json
import zipfile

data_folder = "../data/code-optimization"
data_csv = os.path.join(data_folder, "data.csv")
output_path = os.path.join(data_folder, "data.json")

data_df = pd.read_csv(data_csv)

conversations = []

for i in range(0, len(data_df)):
    unoptimized_code_path = data_df.iloc[i]["Unoptimized Code"]
    optimized_code_path = data_df.iloc[i]["Optimized Code"]
    description = data_df.iloc[i]["Description"]
    
    unoptimized_code = open(os.path.join(data_folder, unoptimized_code_path), "r").read()
    optimized_code = open(os.path.join(data_folder, optimized_code_path), "r").read()
    
    conversation = [
        {"from_agent": 'querry', "value_msg": f"Optimize the givven code\n{unoptimized_code}"},
        {"from_agent": 'response', "value_msg": f"Optimized Code:\n{optimized_code}\n\nSuggestions:\n{description}"},
    ]
    conversations.append({"conversations": conversation})
    
with open(output_path, "w") as f:
    json.dump(conversations, f, indent=4)