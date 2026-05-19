# NYC Crash Frequency-Severity Injury Model

Frequency-severity actuarial model on 2.2M NYC crash records (2012-2025).
using Negative Binomial regression for crash frequency and Gamma GLM for injury severity.

The Bronx in 2012 serves as the baseline. All boroughs 
and year effects are interpreted relative to that reference point.

## Data
NYC Open Data — Motor Vehicle Collisions:
https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes

- Assumed average cost of $35k per injury (medical bills, liabilities, fatalities, property damage, etc.)

## Key Findings
- Brooklyn carries 2x the injury burden of Manhattan
- Post-2020 structural break: frequency fell ~40%, severity doubled
- Net result: only 7% reduction in total expected injuries despite far fewer crashes

## Motivation
As I was studying for Exam P, I encountered many problems involving the Poisson distribution and enjoyed its properties. 
Having prior experience only in forecasting and regression, I wanted to attempt my first
actuarial modeling project. Living in the NYC metro area and having most likely witnessed some of these crashes
firsthand, I built a frequency-severity model to estimate injury burden by borough and
observe how risk changed over time.

This model estimates the economic cost of injuries by borough (not insurance premiums
or payouts) and reveals how risk shifted during and after COVID.

## Limitations
I originally set out to build a pricing model, but encountered factors beyond the current
scope: policy limits, deductibles, insured population counts, and exposure denominators.
The $35K cost assumption is a simplified placeholder to illustrate scale, not a modeled output.

Ideally, this serves as a stepping stone toward a full expected loss model using actual
claims data. The structural break at 2020 is a key signal — rising severity despite fewer
crashes suggests insurers should consider higher reserves for recent policy years.




