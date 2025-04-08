import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = None
# model = "anthropic/claude-3.5-sonnet"

if model is None:
    ratings_json_file = 'ratings.json'
else:
    ratings_json_file = f'ratings_{model.split("/")[-1]}.json'
with open(ratings_json_file, 'r') as f:
    ratings_data = json.load(f)

with open('questions.yml', 'r') as f:
    questions_data = yaml.safe_load(f)

# Convert ratings to DataFrame
df = pd.DataFrame(ratings_data['ratings'])

# Extract model names from subfolder
df['model'] = df['subfolder'].apply(lambda x: x.split('-')[3:])
df['model'] = df['model'].apply(lambda x: ' '.join(x))

# List of questions to analyze
questions = [q['name'] for q in questions_data['questions']]

# Set style and colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Get unique models in a consistent order
unique_models = sorted(df['model'].unique())
model_to_color = dict(zip(unique_models, colors))

# Function to create bar plot for a metric
def plot_metric(data, metric, title, ax, model_to_color):
    # Calculate mean scores for each model
    model_scores = data.groupby('model')[metric].mean().sort_values(ascending=True)

    # Get colors in the same order as the sorted models
    bar_colors = [model_to_color[model] for model in model_scores.index]

    # Create bar plot
    bars = ax.barh(range(len(model_scores)), model_scores, color=bar_colors)

    # Customize plot
    ax.set_yticks(range(len(model_scores)))
    ax.set_yticklabels(model_scores.index)
    ax.set_xlabel('Average Score')
    ax.set_title(title)

    # Add value labels
    for i, v in enumerate(model_scores):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center')

    # Set x-axis limit to 10 (max score)
    ax.set_xlim(0, 10)

# First create overview plot
plt.figure(figsize=(12, 6))
overall_scores = df.groupby('model')['overall_score'].mean().sort_values(ascending=True)
bar_colors = [model_to_color[model] for model in overall_scores.index]
ax = plt.gca()
bars = ax.barh(range(len(overall_scores)), overall_scores, color=bar_colors)
ax.set_yticks(range(len(overall_scores)))
ax.set_yticklabels(overall_scores.index)
ax.set_xlabel('Average Score')
ax.set_title('Overall Scores by Model')
for i, v in enumerate(overall_scores):
    ax.text(v + 0.1, i, f'{v:.2f}', va='center')
ax.set_xlim(0, 10)
plt.tight_layout()
if model is None:
    overall_scores_file = 'overall_scores.png'
else:
    overall_scores_file = f'overall_scores_{model.split("/")[-1]}.png'
plt.savefig(overall_scores_file, dpi=300, bbox_inches='tight')
plt.close()

# Create plots for each question
num_questions = len(questions)
fig, axes = plt.subplots(num_questions//2, 2, figsize=(15, 5*num_questions//2))
axes = axes.flatten()

for i, question in enumerate(questions):
    plot_metric(df, question, f'Scores for {question}', axes[i], model_to_color)

plt.tight_layout()
if model is None:
    question_scores_file = 'question_scores.png'
else:
    question_scores_file = f'question_scores_{model.split("/")[-1]}.png'
plt.savefig(question_scores_file, dpi=300, bbox_inches='tight')
plt.close()
