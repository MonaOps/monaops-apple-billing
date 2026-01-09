import pandas as pd
from google import genai
import os
import webbrowser

# 1. Setup
api_key=os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

TARGET_COMPANY = 'AppleSupport'
CHUNKS_TO_SCAN = 10 # Let's scan the first 1M rows (10 chunks of 100k)

# 2. Process File in Chunks
print(f"🚀 Scanning dataset for {TARGET_COMPANY} using Pure Gemini Logic...")

collected_tweets = []

for i, chunk in enumerate(pd.read_csv('data/twcs.csv', chunksize=100000, low_memory=False)):
    if i >= CHUNKS_TO_SCAN:
        break
        
    # Quick filter using Pandas to find relevant tweets for the AI to look at
    relevant = chunk[chunk['text'].str.contains(TARGET_COMPANY, case=False, na=False)]
    BILLING_TERMS = r"(?:bill|billing|charged|refund|subscription|cancel|trial|payment|invoice|receipt|itunes|app store|apple id|money)"
    relevant = relevant[relevant['text'].str.contains(BILLING_TERMS, case=False, na=False, regex=True)]

    collected_tweets.extend(relevant[['created_at','tweet_id', 'text']].to_dict(orient='records')
)

    print(f"  Processed {i+1}00k rows... Found {len(collected_tweets)} relevant tweets.")

# 3. Consolidate for Gemini
# We'll take the most recent 800 relevant tweets 
# Convert to DataFrame for sorting
df = pd.DataFrame(collected_tweets)

df['created_at'] = pd.to_datetime(
    df['created_at'],
    format='%a %b %d %H:%M:%S %z %Y',
    errors='coerce',
    utc=True
)

df = df.dropna(subset=['created_at'])

# Ensure types are stable
df['tweet_id'] = pd.to_numeric(df['tweet_id'], errors='coerce')
df = df.dropna(subset=['tweet_id', 'created_at'])
df['tweet_id'] = df['tweet_id'].astype('int64')

# Filter to billing candidates (your regex)
BILLING_TERMS = r"(?:bill|billing|charged|refund|subscription|cancel|trial|payment|invoice|receipt|itunes|app store|apple id)"
billing_df = df[df['text'].str.contains(BILLING_TERMS, case=False, na=False, regex=True)].copy()

# Sort deterministically by recency, then ID
billing_df = billing_df.sort_values(['created_at', 'tweet_id'], kind='mergesort')

# Select the most recent 800
final_df = billing_df.tail(800)

print("Selected window:", final_df['created_at'].min(), "→", final_df['created_at'].max())
print("Rows selected:", len(final_df))


tweets_formatted = "\n".join(
    [f"- [{row['created_at'].date()}] {row['text']}" for _, row in final_df.iterrows()]
)


# 4. The "One-Stop" Gemini Prompt
prompt = f"""
I have extracted all customer tweets for {TARGET_COMPANY} from a 2.8M row dataset.
{tweets_formatted}. This dataset contains the most recent 800  customer support tweets (by timestamp) and is used to extract directional signals to inform a decision on billing investment.

Act as a Product Operations analyst supporting a Product Manager decision. and Perform a comprehensive Product Operations Analysis for all tweets related to billing to indicate whether we should prioritise investment in the billing module. 
Goal:
Determine whether we should prioritise product investment in the Billing module now, or whether the signal suggests support-led mitigation / comms / policy clarity is the better lever.

IMPORTANT RULES (must follow):
1) Use ONLY the provided tweets. Do NOT invent data.
2) If you cannot compute a metric from the tweets, output "unknown" and explain what data is missing.
3) For every percentage you provide, also provide the numerator and denominator (counts).
4) Separate "signal" from "inference": clearly label assumptions.
5) Provide confidence (high/medium/low) for each major finding and explain why.
6) Focus ONLY on billing-related tweets. Ignore gratitude-only and unrelated tweets.
7)Show the confidence for all tasks  in triangle/div and highlighted in green if high, yellow if meduim , and red if low and mentione the reason why

Tasks:
A) Billing Root Cause Classification

Purpose:
Extract directional product signals from unstructured customer support tweets related to billing. This classification is interpretive and intended to inform prioritisation decisions, not to establish definitive root cause.

Method:
Apply a consistent text-interpretation rubric to EACH tweet in the provided sample.
Classification must be based solely on the language used in the tweet.
No additional metadata or internal system context is required.

You are explicitly permitted to infer likely root cause from the tweet text.
If a tweet is ambiguous, assign the label "Unknown" and briefly explain the ambiguity.

Root cause rubric (assign exactly ONE per tweet):

1. System failure  
   - Mentions technical errors or failures during billing or subscription actions  
   - Examples: payment failed, error message, charge processed incorrectly, subscription state stuck

2. User error  
   - User explicitly indicates accidental action or missed step  
   - Examples: forgot to cancel, wrong account, child purchase, accidental renewal

3. Policy misunderstanding  
   - Confusion about pricing rules, trials, renewals, or refund eligibility  
   - Examples: trial ended unexpectedly, charged after free trial, refund denied due to policy

4. Confusing UX  
   - Difficulty finding, understanding, or completing billing-related actions  
   - Examples: can’t find cancel option, unclear subscription management, confusing settings

5. Unknown  
   - Insufficient information to confidently distinguish between the above categories

Instructions:
- Classify each tweet in the sample using the rubric above
- After classifying all tweets, compute counts and percentages across the SAMPLE ONLY
- Provide 3–5 representative tweet examples per category
- Clearly state confidence (High / Medium / Low) for the overall distribution and explain why
- Use bar chart to show the data and summarizes the likely root causes of billing-related issues based on the analysis of customer support tweets., also create a table to show the rootcause, count, percentage, exmaple tweets
- Show the confidence in triangle/div and highlighted in green if high, yellow if meduim , and red if low 
- set the backgound colourof this section to #f7e6e6


B) User Segment Signals
- Identify any observable signals of segment (individual/family sharing vs business-managed accounts).
- Provide counts and confidence. If weak, say so.
- set the backgound colourof this section to #e6f7e6

C) Top Pain Points & Severity
- Identify top 3 billing-related pain points by frequency.
- For each: classify as "blocking failure" vs "recoverable friction", and explain reasoning.
- Provide examples.
- set the backgound colourof this section to #f7f7e6
- Explain severity 

D) Churn / Cancellation Intent Signals
- Identify signals of cancellation intent or trust erosion (e.g., “cancel”, “unsubscribe”, “switching”, “chargeback”).
- Provide counts, examples, and confidence.
- Use a table to show the data (signal, count, example tweets)
- set the backgound colourof this section to #e6e6f7

E) Apple-managed vs Third-party
- Estimate proportion of complaints about Apple-managed subscriptions vs third-party services.
- Explain how this changes what we can act on.
- Use a table to show the estimated proportion of complaints (Sybscribtion type , count, and percentage)
- set the backgound colourof this section to #f7e6f7

F) Spikes Over Time (using provided tweet dates)

-Data availability:
Each tweet is prefixed with its creation date in the format [YYYY-MM-DD]. Use these dates.

Instructions:
1) Extract the date from each tweet prefix.
2) Group tweets by ISO week (YYYY-WW).
3) Output a table with: week, tweet_count.
4) Spike rule: mark a week as a "spike" if tweet_count is >= 2x the median weekly tweet_count in this sample.
5) If fewer than 6 distinct weeks exist in the sample, state: "Insufficient time range for reliable spike detection" and still output the weekly counts table.
6) Do NOT say timestamps are required.
7)Use ONLY tweets deemed billing-related (and not gratitude-only) when counting weekly volume
8) Use line chart to show weekly tweet volume, as well as a table to show the week, tweet countm and whether it's a spike or not
9) set the backgound colourof this section to #e6f7f7


G) Decision Recommendation
- Must Output one of: "Prioritise product investment", "Do not prioritise product investment", or "Need more data".
- Must Provide the confidence and the rationale in 5 bullets, key risks, and next 3 actions (engineering/support/product).
- Must use the following CSS:   border: 2px solid #4CAF50;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;

1. If you can’t compute from provided data, mark it Unknown

2. Must show method used (keyword rules, patterns, sampling)

3. Must output counts and percentages only when calculated

The output must end with:

1. Recommend: Invest / Don’t invest / Need more data

2. Confidence: High/Med/Low

3. Next 3 actions



Output:

output Task:
Create a single complete HTML/CSS file titled "MonaOps AI-Powered Insights" that presents the findings clearly for executives. You must use charts and tables when needed.

output format Rules:
1) You MUST NOT add new facts, new numbers, or new conclusions beyond the data in the CSV.
2) Every metric displayed must be taken directly from the CSV.
3) Include a short "Decision Recommendation" section at the top with decision + confidence + next actions.
4) Use Chart.js for charts. Every <canvas> must be wrapped in:
<div style='position: relative; height: 350px; width: 100%; margin-bottom: 20px;'>
5) Chart.js options must set: responsive: true, maintainAspectRatio: false.
6) Output ONLY raw HTML starting with <!DOCTYPE html>.

Return ONLY the raw HTML.

Visual presentation requirements:
- Present key findings using charts and tables wherever quantitative data is available.
- Each major section must be visually separated using spacing and a section header.
- Use a consistent, distinct background or accent color per section to improve scannability.
- Charts should be used for comparisons and trends; tables for exact values.
- Avoid decorative visuals that do not add informational value.
- Make section headers slightly larger or bolder
-Add a thin divider line between major sections

Assumptions & Limitations:
- Analysis is based on a sampled subset of customer support tweets, not the full dataset.
- Classification reflects likely root causes inferred from customer language, not confirmed system diagnoses.
- Tweets often lack full context; ambiguity is expected and captured under "Unknown".
- Results are directional signals intended to guide further investigation, not final decisions.

"""

try:
    print("🤖 Gemini is analyzing and designing the dashboard...")
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=prompt,
        config={
        'temperature': 0.0,  # Forces more deterministic, consistent output
        'top_p': 1,
        'top_k': 1
    }
    )
    
    # Clean and save the HTML
    raw_text = response.text
    clean_html = raw_text.replace('```html', '').replace('```', '').strip()
    
    # Ensure it starts with <!DOCTYPE html> or <html>
    if not clean_html.lower().startswith('<!doctype') and '<html>' not in clean_html.lower():
        # If Gemini missed the tags, we wrap it ourselves
        clean_html = f"<!DOCTYPE html><html><body>{clean_html}</body></html>"
    
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    report_path = os.path.abspath('reports/pure_gemini_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(clean_html)
        
    print(f"✅ Analysis Complete! Opening: {report_path}")
    webbrowser.open(f'file://{report_path}')

except Exception as e:
    print(f"AI Analysis failed: {e}")