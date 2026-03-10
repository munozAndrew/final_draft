## Background

Prediction markets are exchanges where participants buy and sell contracts whose payoffs depend on real world outcomes. The price of a contract represents the crowd's aggregate probability estimate. If a "Yes" contract on a market trades at $0.65, the market is saying there is roughly a 65% chance the event happens. The idea is that competition among informed traders should push prices toward accurate estimates. Anyone who thinks the true probability is higher can profit by buying, and anyone who thinks it is lower can profit by selling.

**Kalshi** is a U.S. regulated prediction market exchange operating under CFTC oversight. It lists binary event contracts across topics like economics, politics, weather, and entertainment. Each market resolves to either "Yes" ($1.00) or "No" ($0.00). Kalshi is a centralized, licensed exchange. You deposit dollars, trade contracts, and withdraw in USD.

**Polymarket** is also a prediction market, but it operates very differently. It runs on the Polygon blockchain using USDC stablecoins and is decentralized and not regulated by U.S. financial authorities. While both are prediction markets answering similar kinds of questions, they are distinct platforms with different legal structures, user bases, and data formats.

This analysis uses Kalshi only, for reasons explained in the Data section.

---

## Hypothesis

Do markets with longer, more complex questions show lower prediction accuracy compared to markets with shorter, simpler ones?

Consider two examples from Kalshi. A question like *"Will CPI rise more than 0.1% in August?"* is 9 words and asks about one number crossing one threshold. A question like *"Will the yield of 10-year U.S. treasury notes be between 3.55 and 3.57 on May 1, 2023?"* is 17 words and includes a specific instrument, a narrow numeric band, and an exact date. The hypothesis is that questions like the second one are harder to price accurately because there are more details a trader can get wrong.

Word count is used here as a **proxy** for complexity. It is not a perfect measure. A short question can still be ambiguous (for example two simultaneous conditions expressed briefly), and a long question can still be precise. This limitation is discussed further below.

---

## Data

### Why Kalshi Only

Polymarket was considered but excluded for several practical reasons. Polymarket has no direct `result` column. To determine what a market resolved to, you have to infer it from settlement prices after the fact, which cannot be used to reconstruct what traders were pricing before the outcome was known. Linking trades back to markets requires matching 77 digit token IDs against a JSON array field in the markets table, which is fragile. Polymarket also has no structured category system. It uses free text questions and URL slugs, so grouping markets by topic would require noisy keyword matching.

Kalshi, by contrast, provides a clean `result` field, a structured `event_ticker` with consistent prefix based categories, and markets stored in parquet files that load without issue.


### Building the Categories

Grouping markets by topic required two steps. First, `extract_categories.py` scanned all Kalshi parquet files and extracted every unique `event_ticker` prefix (the part before the first dash — `FED` from `FED-2024-MAR-25`). This produced 5,562 unique prefixes with market counts and sample titles.

Then `build_domains.py` assigned each prefix to one of four analysis domains using simple prefix-matching rules (e.g., anything starting with `HIGH` → Weather, anything starting with `CPI` or `FED` → Economics / Finance). Prefixes that did not match a domain crypto, rare one-offs were excluded. Sports and parlay markets were also excluded because their titles are auto-generated templates, not real questions.

The four focus domains and market counts used in this analysis:

| Domain | Markets |
|---|---|
| Economics / Finance | 64,350 |
| Weather | 30,650 |
| Politics / Government | 12,078 |
| Entertainment / Awards | 7,049 |

**Total: 114,127 markets**



### Measuring Accuracy: Brier Score

Accuracy is measured using the **Brier Score**:

```
Brier Score = (predicted_probability − actual_outcome)²
```

`predicted_probability` is `last_price / 100`, where `last_price` is the final traded price in cents (0–99). `actual_outcome` is 1 if `result = yes` and 0 if `result = no`. A lower Brier score means more accurate pricing. A market that correctly priced something at 95% scores (0.95 − 1)² = 0.0025. A coin-flip guess scores 0.25.



### Measuring Complexity: Word Count

Complexity is measured as the number of whitespace-separated words in the question title after stripping Kalshi's markdown formatting (`**bold**`). As noted above, word count captures length but not every dimension of complexity. A question with a narrow price band embedded in 8 words can still be harder to price than a vague 15-word question.

---


## Results

### Longer Questions Are Associated With Lower Accuracy

Figure 1 splits all 114,127 focus markets into word-count groups using equal-frequency binning and shows the mean Brier score per group. The shortest questions (3–12 words) have a mean Brier of **0.022**, and the longest (17–81 words) score **0.062** about **2.8× worse**. The Spearman rank correlation between word count and Brier score across all individual markets is **ρ = 0.319 (p < 0.001)**.

<img width="1053" height="651" alt="image" src="https://github.com/user-attachments/assets/4ac2c75d-c47d-4087-b7ca-ae7956a39783" />

One thing visible in Figure 1: the bin for 13-word questions shows a dip (Brier ≈ 0.017). This is likely because a cluster of Politics/Government markets, which are among the most accurately priced, happens to land near this word count. The overall upward trend holds, but it is not perfectly smooth, which is consistent with domain being a confounder.

A note on bin sizes: the bins are targeted at equal-frequency (quintiles), but word counts have many ties at common values like 13 or 14 words. When a percentile boundary falls on a tied value, `pd.qcut` merges bins to avoid empty groups, which is why sample sizes vary across buckets.


### The Domain Breakdown

Figure 2 compares mean Brier score and average question length across the four domains side by side.

<img width="1422" height="588" alt="image" src="https://github.com/user-attachments/assets/2d6108e6-3302-4c42-8535-458579a19b22" />

| Domain | Avg Words | Mean Brier |
|---|---|---|
| Politics / Government | 13.1 | 0.012 |
| Weather | 12.0 | 0.017 |
| Entertainment / Awards | 11.0 | 0.042 |
| Economics / Finance | 14.9 | 0.043 |

The most interesting case is **Entertainment / Awards**: it has the *shortest* average questions (11 words) but the second-worst accuracy (Brier ≈ 0.042), nearly tied with Economics / Finance. This does not fit the simple "longer = worse" pattern at the domain level, and it is worth being honest about. Entertainment outcomes (who wins an Oscar, which song tops the chart) are inherently harder to predict regardless of how the question is phrased. Domain-specific predictability appears to be a real factor that word count alone does not capture.

**Economics / Finance** has the longest questions (14.9 words) and the worst accuracy (Brier ≈ 0.043), consistent with the hypothesis. **Weather** questions are shorter (12.0 words) and more accurate (Brier ≈ 0.017), likely because public weather forecasts give traders a reliable anchor. **Politics / Government** is the outlier on the other end: moderate question length (13.1 words) but the best accuracy (Brier ≈ 0.012), probably because most political questions track slowly-moving, heavily-reported numbers like approval ratings.


### Category-Level Pattern

Figure 3 plots each specific market category (event ticker prefix) as a dot, with mean word count on the x-axis and mean Brier on the y-axis. The OLS trend line has a positive slope (0.0024 per word), but the relationship is not statistically significant at conventional thresholds (p = 0.087). This means that while the trend is in the expected direction, it is not strong enough at the category level to rule out sampling noise.

<img width="1050" height="650" alt="image" src="https://github.com/user-attachments/assets/d9fa51a7-409b-4f3e-b496-ab13c8cf4cc2" />


 Controlling for Domain: Multivariate OLS

One concern with the bivariate relationship is that domain is a confounder: Economics / Finance has both the longest questions *and* the hardest-to-predict outcomes, so the word count effect could just be picking up domain membership rather than question length.

To check this, a multivariate OLS model was estimated:

```
Brier ~ word_count + domain (as dummy variables)
```

The reference category is Economics / Finance. Results (n = 114,127, R² = 0.013):

| Variable | Coef | SE | t | p |
|---|---|---|---|---|
| Intercept | 0.0262 | 0.0020 | 13.08 | < 0.001 |
| word\_count | **0.00111** | 0.00013 | 8.51 | **< 0.001** |
| Entertainment / Awards | +0.0031 | 0.0016 | 2.00 | 0.045 |
| Politics / Government | −0.0288 | 0.0012 | −24.35 | < 0.001 |
| Weather | −0.0227 | 0.0009 | −25.44 | < 0.001 |

The `word_count` coefficient remains positive and statistically significant (β = 0.00111, p < 0.001) even after controlling for domain. Each additional word in a question title is *associated* with a Brier score increase of roughly 0.001, a small effect per word, but meaningful across the range of word counts in this dataset. The large domain effects (Politics / Government and Weather both significantly lower than Economics / Finance) confirm that domain matters a lot, but word count has an independent association beyond that.

The overall R² = 0.013 is low, meaning this model explains only about 1.3% of the variance in Brier scores. Word count and domain together are not nearly the whole story.

### OLS Diagnostics

Figure 4 shows the standard diagnostic plots for the multivariate OLS model.

<img width="893" height="644" alt="image" src="https://github.com/user-attachments/assets/fbdb9783-db2a-4f36-9e75-052ff9c0c0b0" />

The residuals vs fitted plot shows a characteristic pattern: residuals fan out at low fitted values. This is expected because Brier scores are bounded at 0 and right-skewed by construction, markets priced near certainty can't have large errors, while markets priced near 50% can. The QQ plot and histogram confirm that residuals are not normally distributed; they are right-skewed. This means the OLS p-values should be interpreted carefully, though with n = 114,127, the central limit theorem makes the t-statistics reasonably reliable. The scale-location plot shows heteroscedasticity, which is also expected given the bounded, skewed outcome.

Figure 5 shows Cook's distance and standardized residuals for the category-level bivariate OLS, where n is small enough for these diagnostics to be interpretable.

<img width="1421" height="596" alt="image" src="https://github.com/user-attachments/assets/42c1f467-6e8c-430c-a7e2-28c1949c4009" />

---
## Discussion

The core finding is that longer question titles are associated with worse prediction accuracy, and this relationship holds at the individual market level (ρ = 0.319) and persists in a multivariate model that controls for domain (β = 0.00111, p < 0.001). At the category level, the trend is in the expected direction but is not statistically significant (p = 0.087), which is a real limitation worth noting.

Two mechanisms might explain the individual market association. First, **misreading**: a question that specifies a narrow price band, a particular instrument, and an exact date gives traders more ways to get a detail wrong. Second, **thin participation**: complex questions may attract fewer confident traders, so pricing errors do not get corrected as quickly.

The Entertainment / Awards domain complicates the story at the domain level. Short questions do not guarantee accurate markets, the underlying topic also has to be predictable. This suggests the relationship between question length and accuracy is real but moderate in size, and domain specific factors like inherent outcome uncertainty may be just as important.

### Limitations

**Word count is a rough proxy.** A short question can still embed multiple conditions or require specialized knowledge. A long question can still be easy to price if the reference data is public and updated frequently.

**Causality is not established.** This is purely observational analysis. The association between question length and Brier score is consistent with several possible explanations (traders misread long questions, long questions attract fewer participants, complex topics generate complex questions), but the data cannot distinguish between them.

**`last_price` timing.** The final trade on a market can happen after the outcome is already known but before Kalshi officially resolves it. For narrow threshold markets, the price may snap to near 0 or 100 as soon as the reference data is published. That would make those markets appear more accurate in hindsight than they were when uncertainty was genuine.

**Low R².** The model explains only 1.3% of Brier score variance. Question length and domain together are a small part of what drives prediction accuracy.
