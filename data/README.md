# ⚽ FIFA World Cup 2026 — Monte Carlo Win Probability Dataset

**All 48 Qualified Teams | Real FIFA Rankings (Jan 19, 2026) | 50,000 Monte Carlo Simulations**

---

## 📁 Files

| File | Description |
|------|-------------|
| `fifa_wc2026_dataset.csv` | Main dataset — 48 teams × 25 columns |
| `column_weights.csv` | Weight of each feature in the strength model |
| `dataset-metadata.json` | Full metadata and simulation details |

---

## 🏆 Top 10 Win Probabilities

| Rank | Team | FIFA Rank | Win % | Final % |
|------|------|-----------|-------|---------|
| 1 | Brazil | #5 | **24.8%** | 39.4% |
| 2 | Argentina | #2 | **20.5%** | 35.0% |
| 3 | Germany | #10 | **20.1%** | 34.7% |
| 4 | France | #3 | **16.8%** | 30.4% |
| 5 | Spain | #1 | **7.0%** | 16.9% |
| 6 | England | #4 | **4.0%** | 11.6% |
| 7 | Mexico | #16 | **2.3%** | 8.2% |
| 8 | Netherlands | #7 | **1.1%** | 4.8% |
| 9 | United States | #15 | **1.0%** | 4.6% |
| 10 | Uruguay | #17 | **0.8%** | 3.9% |

---

## 📊 Columns

### Input Features (Real Data)
| Column | Type | Source |
|--------|------|--------|
| `fifa_ranking_jan2026` | int | Official FIFA — Jan 19, 2026 |
| `fifa_points_jan2026` | int | Official FIFA — Jan 19, 2026 |
| `world_cup_titles` | int | Historical fact |
| `world_cup_finals` | int | Historical fact |
| `world_cup_appearances` | int | Historical fact |
| `star_player_rating` | float (0-10) | Expert estimation |
| `avg_player_age` | float | Squad analysis |
| `goalkeeper_rating` | float (0-10) | Expert estimation |
| `squad_depth_score` | float (0-10) | Expert estimation |
| `coach_experience_tournaments` | int | Expert estimation |
| `h2h_vs_top10_winrate` | float (0-1) | Historical estimation |
| `knockout_stage_reach_rate` | float (0-1) | Historical WC data |
| `is_host` | int (0/1) | Official — USA, Canada, Mexico |

### Calculated Features
| Column | Formula |
|--------|---------|
| `historical_score` | `(titles×15) + (finals×6) + (apps×1.5) + (ko_rate×12)` |
| `squad_score` | `star×0.35 + gk×0.25 + depth×0.25 + coach×0.15` (normalized) |
| `composite_strength` | Weighted sum of all features (0–100 scale) |

### Monte Carlo Output (Target Variables)
| Column | Description |
|--------|-------------|
| `mc_group_stage_pct` | % simulations team advanced from groups |
| `mc_r16_pct` | % reached Round of 16 |
| `mc_quarterfinal_pct` | % reached Quarter-Final |
| `mc_semifinal_pct` | % reached Semi-Final |
| `mc_final_pct` | % reached the Final |
| `mc_win_probability_pct` | % won the tournament (**all 48 sum to ~100%**) |

---

## ⚙️ Model Details

### Feature Weights
| Feature | Weight | Reason |
|---------|--------|--------|
| FIFA Points | **28%** | Most current official data |
| Star Player Rating | **15%** | Individual brilliance in knockouts |
| GK Rating | **12%** | Most impactful position — shootouts |
| Historical Score | **12%** | World Cup DNA and pedigree |
| Squad Depth | **10%** | 7 matches over 30 days |
| H2H vs Top-10 | **8%** | Proven record vs elite |
| Coach Experience | **7%** | Tactical expertise |
| Average Age | **5%** | Peak age ~27.5 |
| Host Bonus | **3%** | Home advantage |

### Monte Carlo Simulation
```
Simulations: 50,000 full tournaments
Match model: P(A beats B) = 1 / (1 + exp(-(sA - sB) / 6.0))
Format: 12 groups of 4 → 32 advance → R16 → QF → SF → Final
Draw rate: ~30% in group stage
Host boost: ×1.08 strength multiplier
```

---

## 💡 Suggested Use Cases
- **Win probability prediction** — use `mc_win_probability_pct` as regression target
- **Feature importance analysis** — which features drive tournament success?
- **Classification** — predict group stage advancement (binary target: `mc_group_stage_pct > 50`)
- **Clustering** — group teams by playing profile and confederation strength
- **Comparison** — how do statistical models compare to betting odds?

---

## ⚠️ Notes
- 6 spots marked `is_tbd=1` (4 UEFA playoffs + 2 intercontinental playoffs — March 2026)
- FIFA Rankings next update: **April 1, 2026** (post-playoffs)
- Squad quality ratings are expert estimations, not official data

---

## 📌 Data Sources
- **FIFA Rankings**: January 19, 2026 official release (ESPN / FIFA Coca-Cola Ranking)
- **WC History**: Verified historical records
- **Simulation**: Custom Python Monte Carlo implementation

**License**: CC BY 4.0 — Free to use with attribution
