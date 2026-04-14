import matplotlib.pyplot as plt
import numpy as np

# Error counts
categories = ['Boundary Error', 'Missing Entity', 'Incorrect Extraction', 'Incorrect Entity Type']
values = [7, 5, 2, 0]

# Split categories into two lines for better readability
categories_multiline = ['Boundary\nError', 'Missing\nEntity', 'Incorrect\nExtraction', 'Incorrect\nEntity Type']

# Create horizontal bar plot
fig, ax = plt.subplots(figsize=(10, 4))
y_pos = np.arange(len(categories))

# Sort by value (largest at bottom)
sorted_indices = np.argsort(values)[::-1]  # Reverse to put largest at bottom
sorted_categories = [categories_multiline[i] for i in sorted_indices]
sorted_values = [values[i] for i in sorted_indices]

bars = ax.barh(y_pos, sorted_values, color='#2E7AB4', height=0.5, edgecolor='none', linewidth=0)

ax.set_xlabel('Count', fontsize=15, fontweight='bold')
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_categories, fontsize=15, fontweight='bold')
ax.tick_params(axis='x', labelsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

plt.tight_layout()
plt.savefig('error_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved to error_distribution_analysis.png")
