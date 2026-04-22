
# Tree vs. Linear Disagreement Analysis

## Sample Details

* **Test-set index:** 777
* **True label:** 0
* **RF predicted P(churn=1):** 0.5998
* **LR predicted P(churn=1):** 0.1700
* **Probability difference:** 0.4299

## Feature Values

* **tenure:** 36.0
* **monthly_charges:** 20.0
* **total_charges:** 1077.33
* **num_support_calls:** 2.0
* **senior_citizen:** 0.0
* **has_partner:** 0.0
* **has_dependents:** 0.0
* **contract_months:** 1.0

## Structural Explanation

The random forest captured the interaction between `num_support_calls ≥ 2` and
`contract_months = 1` (month-to-month): customers who have contacted support multiple
times on a short-term contract sit in a high-risk branch regardless of their very low
`monthly_charges` ($20). The tree first splits on `contract_months == 1`, and inside
that branch splits again on `num_support_calls ≥ 2`, producing a conjunctive rule that
neither feature triggers alone. Logistic regression can only weight each feature
independently — it combines the strong downward signal from low monthly charges with the
individual (weaker) signals from support calls and contract type, arriving at a much
lower probability of 0.17. This specific pattern — where two moderate-risk features
together create elevated risk — is a feature interaction that a linear model structurally
cannot express without explicit interaction terms added as engineered features.
