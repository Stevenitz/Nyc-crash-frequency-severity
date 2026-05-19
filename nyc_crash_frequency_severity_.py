
"""
Created on Mon Jan 26 20:49:59 2026

@author: Stevenitzz
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
from scipy import stats

Crash= pd.read_csv(r'C:\Motor_Vehicle_Collisions_-_Crashes.csv',parse_dates=['CRASH DATE'])

print(f"New data shape: {Crash.shape}")
print(f"Columns: {Crash.columns.tolist()}")

# Columns I found Interesting
keep_columns = [
    'CRASH DATE',
    'CRASH TIME', 
    'BOROUGH',
    'NUMBER OF PERSONS INJURED',
    'NUMBER OF PERSONS KILLED',
    'NUMBER OF PEDESTRIANS INJURED',
    'NUMBER OF PEDESTRIANS KILLED',
    'NUMBER OF CYCLIST INJURED',
    'NUMBER OF CYCLIST KILLED',
    'NUMBER OF MOTORIST INJURED',
    'NUMBER OF MOTORIST KILLED',
    'CONTRIBUTING FACTOR VEHICLE 1',
    'VEHICLE TYPE CODE 1'
]

# Filter to just these columns
Crash = Crash[keep_columns].copy()
print(f"Cleaned shape: {Crash.shape}")


# Extract year from CRASH DATE
Crash["year"] = Crash["CRASH DATE"].dt.year


Crash["BOROUGH"] = Crash["BOROUGH"].fillna("Unknown")



Crash["VEHICLE TYPE CODE 1"].value_counts().head(10)
Crash["CONTRIBUTING FACTOR VEHICLE 1"].value_counts().head(10)


injury_cols = [
    "NUMBER OF PERSONS INJURED",
    "NUMBER OF PERSONS KILLED",
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED"
]

## DATA CLEAN UP
Crash[injury_cols] = Crash[injury_cols].fillna(0)


Crash['total_injuries'] = Crash['NUMBER OF PERSONS INJURED'] + Crash['NUMBER OF PERSONS KILLED']

Crash.info()
Crash.head()
Crash.describe()

Crash['date'] = pd.to_datetime(Crash['CRASH DATE'])
daily_counts = Crash.groupby(['BOROUGH', 'date']).size().reset_index(name='daily_crashes')


daily_counts['year'] = daily_counts['date'].dt.year
daily_counts['month'] = daily_counts['date'].dt.month
daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
daily_counts['is_weekend'] = (daily_counts['day_of_week'] >= 5).astype(int)

# Use ALL complete years 
model_data = daily_counts[daily_counts['year'].between(2012, 2025)].copy()

#  Frequency Model (Negative Binomial)

from statsmodels.discrete.discrete_model import NegativeBinomial


#simple model on purpose 
freq_model2 = NegativeBinomial.from_formula('daily_crashes ~ C(BOROUGH) ',data=model_data).fit()

model3=NegativeBinomial.from_formula('daily_crashes ~ C(BOROUGH) + C(year)',data=model_data).fit()


print(freq_model2.summary())
print(model3.summary())   # winner has double the explaing power



#residuals log scale funnel is fine
# Residual diagnostics

# Residuals vs Fitted



fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Plot 1: Residuals vs Fitted
axes[0].scatter(model3.fittedvalues, model3.resid, alpha=0.3)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Plot 2: QQ Plot
stats.probplot(model3.resid, dist="norm", plot=axes[1])
axes[1].set_title('QQ Plot')

plt.tight_layout()
plt.show()



# Model selection Metrics 

print("Min predicted:", model3.fittedvalues.min())
print("Max predicted:", model3.fittedvalues.max())
print("Actual daily crashes range:", model_data['daily_crashes'].min(), "to", model_data['daily_crashes'].max())





# Find the day with largest residual
max_resid_idx = model3.resid.idxmax()
outlier_day = model_data.iloc[max_resid_idx]
print("Outlier day:", outlier_day[['BOROUGH', 'date', 'daily_crashes']])
print("Predicted:", model3.fittedvalues.iloc[max_resid_idx])
print("Residual:", model3.resid.iloc[max_resid_idx])




from statsmodels.tools.eval_measures import aic, bic

# After fitting models, compare them systematically
model_names = ['Borough Only', 'Borough + Year']
models = [freq_model2, model3]

for name, model in zip(model_names, models):
    print(f"\n{name}:")
    print(f"AIC: {model.aic:.2f}, BIC: {model.bic:.2f}")
    print(f"LLR p-value: {model.llr_pvalue:.4f}")
    print(f"Log-Likelihood: {model.llf:.2f}")


# create injury variables

# Filter to years 2012-2025 BEFORE aggregating
Crash_filtered = Crash[Crash['year'].between(2012, 2025)].copy()

#  create daily injuries from filtered data
daily_injuries = Crash_filtered.groupby(['BOROUGH', 'date'])['total_injuries'].sum().reset_index()
daily_injuries = daily_injuries.rename(columns={'total_injuries': 'daily_injuries'})

model_data = daily_counts.merge(daily_injuries, on=['BOROUGH', 'date'], how='left')
model_data['daily_injuries'] = model_data['daily_injuries'].fillna(0)




#NEW GAMMA
# For severity model, only use crashes WITH injuries
severity_data = model_data[model_data['daily_injuries'] > 0].copy()




# Severity = injuries per crash (only for crashes with injuries)
#(injuries / crash)
severity_data['severity'] = (
    severity_data['daily_injuries'] / severity_data['daily_crashes']
)


# Gamma
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import log

severity_model = smf.glm(
    'severity ~ C(BOROUGH) + C(year)',
    data=severity_data,
    family=Gamma(link=log())
).fit()


print(severity_model.summary())

 

# Only predict for years the model was trained on
model_data_pred = model_data[model_data['year'].between(2012, 2025)].copy()

# Now predict

model_data_pred['predicted_daily_crashes'] = model3.predict(model_data_pred)
model_data_pred['predicted_severity'] = severity_model.predict(model_data_pred)

model_data_pred['pure_premium'] = (
    model_data_pred['predicted_daily_crashes'] * 
    model_data_pred['predicted_severity']
)



print(model_data_pred[['BOROUGH', 'year', 'predicted_daily_crashes', 
                           'predicted_severity', 'pure_premium']].head())


yearly_results = model_data_pred.groupby(['BOROUGH', 'year']).agg(
    total_predicted_crashes=('predicted_daily_crashes', 'sum'),
    avg_severity=('predicted_severity', 'mean'),
    total_predicted_injuries=('pure_premium', 'sum')
).reset_index()

print(yearly_results.head(20))

#easil see severity increase 

#$ cost

# Example assumptions (simplified for illustration)
cost_per_minor_injury = 15000    # Medical + pain & suffering
cost_per_major_injury = 75000    # Serious injuries
cost_per_fatality = 500000       # Fatalities

# Since I don't have injury severity breakdown, use weighted average
# data pattern: mostly minor, some serious, few fatalities
avg_cost_per_injury = 35000  # Reasonable placeholder:
    
    
yearly_results['expected_dollar_losses'] = yearly_results['total_predicted_injuries'] * avg_cost_per_injury

# Add exposure if you had it (number of insured cars)
# yearly_results['pure_premium_per_car'] = yearly_results['expected_dollar_losses'] / number_of_cars

print(yearly_results[['BOROUGH', 'year', 'total_predicted_injuries', 
                      'expected_dollar_losses']].head())

# example bx 2013 =228.2 million

# Pivot table to compare boroughs side by side
###NOTE WERE BACK INJURIES?




pivot_table = yearly_results.pivot_table(
    index='BOROUGH',
    columns='year',
    values='total_predicted_injuries',
    aggfunc='sum'
)
print(pivot_table)

# for specific year:
    # Show all boroughs for a specific year (e.g., 2023)
year_2020 = yearly_results[yearly_results['year'] == 2020]
print(year_2020[['BOROUGH', 'total_predicted_crashes', 'avg_severity', 'total_predicted_injuries']])
    



year_2025 = yearly_results[yearly_results['year'] == 2025]
print(year_2025[['BOROUGH', 'total_predicted_crashes', 'avg_severity', 'total_predicted_injuries']])








#HEATMAP/VISUALS for port mostly 

#1. premiums by borough and year- more honest

# Pivot your yearly_results for heatmap
premium_matrix = yearly_results.pivot(
    index='BOROUGH',
    columns='year', 
    values='total_predicted_injuries'  # or 'expected_dollar_losses'
)

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(premium_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
            linewidths=0.5, cbar_kws={'label': 'Expected Injuries'})
plt.title('Expected Injuries by Borough and Year')
plt.xlabel('Year')
plt.ylabel('Borough')
plt.tight_layout()
plt.show()


#by $ amounts ezier to interpet cause no assumption andis my model and shows actual work
# Add dollar conversion first
yearly_results['expected_dollar_losses'] = yearly_results['total_predicted_injuries'] * 35000

dollar_matrix = yearly_results.pivot(
    index='BOROUGH',
    columns='year',
    values='expected_dollar_losses'
)

plt.figure(figsize=(12, 8))
sns.heatmap(dollar_matrix/1e6, annot=True, fmt='.1f', cmap='RdYlGn_r',
            linewidths=0.5, cbar_kws={'label': 'Million $'})
plt.title('Expected Losses by Borough and Year (Millions $)')
plt.xlabel('Year')
plt.ylabel('Borough')
plt.tight_layout()
plt.show()




#see our errors in ^

# Calculate what % of crashes have injuries
injury_rate = (model_data['daily_injuries'] > 0).mean()
print(f"Overall injury rate: {injury_rate:.1%}")

# By borough
injury_rates_by_borough = model_data.groupby('BOROUGH').apply(
    lambda x: (x['daily_injuries'] > 0).mean()
)
print(injury_rates_by_borough)
print((model_data['daily_injuries'] > 0).mean())




# Gamma model residuals
plt.scatter(severity_model.fittedvalues, severity_model.resid_deviance)
plt.axhline(y=0, color='r')
plt.title('Gamma Model Deviance Residuals')

print(yearly_results.groupby('BOROUGH')['total_predicted_injuries'].sum().round(0))

pre = yearly_results[yearly_results['year'] < 2020]['total_predicted_injuries'].mean()
post = yearly_results[yearly_results['year'] >= 2020]['total_predicted_injuries'].mean()
print((post - pre) / pre)

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Year coefficients both models ---
years = list(range(2013, 2026))

freq_coefs = [0.0004,-0.0143,0.0304,0.0471,0.0284,0.0412,
              -0.0720,-0.6809,-0.6951,-0.7491,-0.8170,-0.8625,-0.8897]
sev_coefs  = [-0.0033,-0.0797,-0.1298,-0.0711,-0.0699,-0.0737,
               0.0843, 0.3870, 0.5053, 0.5761, 0.6883, 0.7401, 0.7426]

axes[0].plot(years, freq_coefs, marker='o', label='Frequency (NB)', color='steelblue')
axes[0].plot(years, sev_coefs,  marker='s', label='Severity (Gamma)', 
             color='tomato', linestyle='--')
axes[0].axvline(x=2020, color='gray', linestyle=':', label='COVID break')
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('Year Coefficients: Frequency vs Severity')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Coefficient (log scale)')
axes[0].legend()

# --- Right: Borough coefficients ---
boroughs = ['Bronx\n(base)', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Is.', 'Unknown']
freq_b = [0, 0.7664, 0.3376, 0.5716, -1.2951, 1.0653]
sev_b  = [0, 0.0162, -0.3164, -0.0685, -0.0211, 0.0768]

x = np.arange(len(boroughs))
width = 0.35
axes[1].bar(x - width/2, freq_b, width, label='Frequency (NB)', color='steelblue')
axes[1].bar(x + width/2, sev_b,  width, label='Severity (Gamma)', color='tomato')
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(boroughs, fontsize=9)
axes[1].set_title('Borough Coefficients: Frequency vs Severity')
axes[1].set_ylabel('Coefficient (log scale)')
axes[1].legend()

plt.suptitle('Frequency-Severity Model Coefficients', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('coef_comparison.png', dpi=150, bbox_inches='tight')
plt.show()