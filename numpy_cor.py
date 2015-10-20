"""
In this script:
1. Correlation

2. Multiple Variable Regression for a basic forecast model

3. Loop that automates a process-of-elimination to eliminate 
variables that have p_values that are too high, for a strong forecast model:
(High R-squared)

About the data: 
First column of this data file was date strings, so I sliced away.
Second column was the dependent variable I wanted to test the others against.
"""

import csv
import numpy as np
import statsmodels.api as sm
import pprint

pp = pprint.PrettyPrinter(indent=1)

# Open up csv
with open('file_path','r') as f:
	reader = csv.reader(f)
	# Save data in memory
	data = [row for row in reader]
	column_titles = data[0]

# Column titles for reference:
#for item in column_titles:
# 	print ('{} {}'.format(column_titles.index(item), item))

# .T transposes data for the appropriate calc
a = np.array(data[1:], dtype=object).T

# a[0] was date_strings. Not needed. So:
data_floats = a[1:].astype(float)


# 1. Correlation coefficients.
corr_coefs = np.corrcoef(data_floats)

# Now I want to print significant correlations, 
# without repeating ones that appear in the symmetrical output.
sig_corr_coefs = []
for column in range(len(corr_coefs)):
	for coef in range(column,len(corr_coefs)):
		if corr_coefs[column][coef] > .5 or corr_coefs[column][coef] < -.5:
			# Don't keep the 1:1 correlation of a variable against itself
			if column_titles[1:][column] != column_titles[1:][coef]:
				sig_corr_coefs.append([
					column_titles[1:][column],
					column_titles[1:][coef],
					corr_coefs[column][coef]
				])


print("Significant Correlation Coefficients:")
for line in sig_corr_coefs:
	print(line)


# 2. Multiple Regression from statsmodels docs
y = data_floats[0]
x = data_floats[1:].T

capital_x = sm.add_constant(x)
model = sm.OLS(y, capital_x)
fit = model.fit()
fit.summary()

ps = fit.pvalues[1:]
print(ps)

# Find index for largest p_value
max_index = np.argmax(ps)

# Pop that metric, if it's above 5%. Save a list of the eliminated
columns_kept = column_titles[2:]
columns_elim = []

if np.amax(ps) > .05:
	columns_elim.append(
		{'title': columns_kept.pop(max_index),
		'p_value': np.amax(ps)}
	)
	x = np.delete(x,max_index,1)


# 3. While-loop to automate process-of-elimination until all p_values are less than 5%

y = data_floats[0]
x = data_floats[1:].T

max_p_value = .05
columns_kept = column_titles[2:]
columns_elim = []

while max_p_value >= .05:
	capital_x = sm.add_constant(x)
	model = sm.OLS(y, capital_x)
	fit = model.fit()
	fit.summary()

	# Find index for largest p_value
	ps = fit.pvalues[1:]
	max_index = np.argmax(ps)

	# Pop that metric if it's above 5%, save a list of the eliminated
	max_p_value = np.amax(ps)
	if max_p_value > .05:
		columns_elim.append(
			{'title': columns_kept.pop(max_index),
			'p_value': max_p_value}
		)
		x = np.delete(x, max_index, 1)

# print('Eliminated items:')
# pp.pprint(columns_elim)
print(fit.summary())

print('Model items:')
for i in range(len(ps)):
	print('{}: P value = {}%'.format(columns_kept[i], round(ps[i]*100,4)))

# Store for forecast:
forecast_model = [
	# Start with constant
	{'name':'constant','coef': fit.params[0]}
]
for i in range(len(ps)):
	forecast_model.append({
		'name': columns_kept[i],
		'coef': fit.params[i+1]
	})