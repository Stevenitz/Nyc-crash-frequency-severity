# NYC Crash Frequency-Severity Injury Model

Frequency-severity actuarial model on 2.2M NYC crash records (2012-2025).
using Negative Binomial regression for crash frequency and Gamma GLM for injury severity.

The Bronx in 2012 serves as the baseline. All boroughs 
and year effects are interpreted relative to that reference point.

# v2: Upgraded with new time features + 2026 out-of-sample validation.

## Data
NYC Open Data — Motor Vehicle Collisions:
https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes

- Assumed average cost of $35k per injury (medical bills, liabilities, fatalities, property damage, etc.)

## Key Findings
- Post-2020 structural break: daily crash frequency fell 49.5%, while injury severity per crash increased 59.1%.
- Net result: total expected injuries declined only 19.7% despite far fewer crashes
- 2026 out-of-sample validation: MAE 6.83, RMSE 9.32 injuries/day

## Motivation
As I was studying for Exam P, I encountered many problems involving the Poisson distribution and enjoyed its properties. 
Having prior experience only in forecasting and regression, I wanted to attempt my first
actuarial modeling project. Living in the NYC metro area and having most likely witnessed some of these crashes
firsthand, I built a frequency-severity model to estimate injury burden by borough and
observe how risk changed over time.

This model estimates the economic cost of injuries by borough (not insurance premiums
or payouts) and reveals how risk shifted during and after COVID.

## Limitations
The data itself was incomplete. A large portion of records had no borough assigned 
and I grouped these as "Unknown" and kept them in the model since removing them would 
have biased my results given how prevalent they were.

I originally set out to build a pricing model but ran into factors beyond my current 
scope: policy limits, deductibles, insured population counts, and exposure denominators. 
The $35K cost assumption is a rough placeholder to illustrate scale, not a model output.

Ideally, this is a stepping stone toward a full expected loss model using actual claims 
data. The structural break at 2020 was the most important thing I found: rising severity 
despite fewer crashes is a signal that models built on older assumptions may no longer 
hold, and that we have to be ready to update them when the underlying risk changes.




I am currently sitting for MAS-I, hoping to improve my statistical methods, and will recreate this project in R with additional features.



