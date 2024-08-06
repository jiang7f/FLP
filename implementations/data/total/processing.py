import pandas as pd

file_path = '_depth.csv'
df = pd.read_csv(file_path)

# df = df.drop(columns=['pbid'])

grouped_df = df.groupby(['pkid', 'layers', 'method',], as_index=False).agg({
    'depth': 'mean',
    'culled_depth': 'mean',
})

new_path = f'{file_path[:-4]}_processed.csv'
grouped_df.to_csv(new_path, index=False)

print(f'to {new_path}')
