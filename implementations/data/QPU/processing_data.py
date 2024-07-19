import pandas as pd

file_path = 'qpu_evaluate.csv'
df = pd.read_csv(file_path)

# df = df.drop(columns=['pbid'])

grouped_df = df.groupby(['pkid', 'layers', 'method', 'backend', 'shots'], as_index=False).agg({
    'ARG': 'mean',
    'in_constraints_probs': 'mean',
    'best_solution_probs': 'mean',
    'iteration_count': 'mean'
})

new_path = f'processed_{file_path}'
grouped_df.to_csv(new_path, index=False)

print(f'to {new_path}')
