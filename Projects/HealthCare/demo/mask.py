import os
import json
import time

import numpy as np
import pandas as pd

from tqdm import tqdm
from openai import OpenAI

if __name__ == '__main__':
    data_dir = 'data'
    df = pd.read_excel(os.path.join(data_dir, f'data.xlsx'))

    client = OpenAI(api_key='YOUR_API_KEY_HERE')
    model_name = 'gpt-4o'
    
    prompt = "Mask the date-related private information and return the masked clinic note: {}"
    responses = []
    total_gen_num = 0
    once_time = time.time()
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        total_gen_num += 1
        question = prompt.format(row['note'])
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}]
        
        try:
            per_start_time = time.time()
            response = client.chat.completions.create(model = model_name,
                                                      messages = messages)
            content = response.choices[0].message.content
        except Exception as e:
            print('Error:', e)
            print('Number of', total_gen_num, 'not generated')
            content = np.nan
        
        if total_gen_num % 10 == 0:
            print('Current execution time of ', total_gen_num, ' : ', time.time() - once_time)
            once_time = time.time()
            time.sleep(10)

        df.at[index, f'masked'] = content
        print(content)
    df.to_excel(os.path.join(data_dir, 'results.xlsx'), index=False)
    