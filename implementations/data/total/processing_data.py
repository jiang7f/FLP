import pandas as pd
import ast

# Helper function to parse the list and get the max value
def get_max_from_list(cell):
    return max(ast.literal_eval(cell))
    # return sum(ast.literal_eval(cell)) / len(ast.literal_eval(cell))

# Helper function to parse the list and convert to numeric
def iteration(cell):
    return sum([int(x) for x in ast.literal_eval(cell)])

# Load the data
file_path = 'evaluate_G_dich.csv'
df = pd.read_csv(file_path)

# Drop the 'pbid' column
df = df.drop(columns=['pbid'])

# Apply the helper function to the specified columns
df['ARG'] = df['ARG'].apply(get_max_from_list)
df['in_constraints_probs'] = df['in_constraints_probs'].apply(get_max_from_list)
df['best_solution_probs'] = df['best_solution_probs'].apply(get_max_from_list)
df['iteration_count'] = df['iteration_count'].apply(iteration)

# Group by the specified columns and calculate the mean
grouped_df = df.groupby(['pkid', 'layers', 'variables', 'constraints', 'method'], as_index=False).agg({
    'ARG': 'mean',
    'in_constraints_probs': 'mean',
    'best_solution_probs': 'mean',
    'iteration_count': 'mean'
})

# Save the processed DataFrame to a new CSV file
new_path = f'processed_{file_path}'
grouped_df.to_csv(new_path, index=False)

print(f'to {new_path}')
