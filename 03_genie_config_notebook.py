# Databricks notebook source
# MAGIC %md
# MAGIC # Wyndham Revenue Management Genie Space - Notebook 03
# MAGIC ## Genie Configuration and Instructions Setup
# MAGIC 
# MAGIC **Project**: Wyndham Hotels & Resorts AI/BI Genie Space  
# MAGIC **Focus**: Revenue Management & Pricing Intelligence  
# MAGIC **Phase**: Genie Space Configuration and Instructions  
# MAGIC **Dependencies**: Notebooks 01-02 (Environment Setup & Data Generation)
# MAGIC 
# MAGIC ### Objectives
# MAGIC 1. **Create optimized materialized views** for Genie performance
# MAGIC 2. **Configure Genie space** with proper data sources and scope
# MAGIC 3. **Implement comprehensive instructions** with revenue management business logic
# MAGIC 4. **Establish benchmark questions** for accuracy validation
# MAGIC 5. **Test and validate** Genie responses against known correct answers
# MAGIC 
# MAGIC ### Prerequisites Validation
# MAGIC - ✅ Unity Catalog schemas created (Notebook 01)
# MAGIC - ✅ Synthetic data generated and validated (Notebook 02)
# MAGIC - ✅ 900 properties across 8 brands (US + Canada)
# MAGIC - ✅ 1,029,300 daily performance records (2021-2023)
# MAGIC - ✅ 399,973 guest transactions with realistic patterns
# MAGIC - ✅ RevPAR calculation errors resolved (0 errors remaining)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: Environment Validation and Setup

# COMMAND ----------

# Import required libraries
import json
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Set catalog and schema context
catalog = "main"
staging_schema = "wyndham_staging"
curated_schema = "wyndham_curated"
analytics_schema = "wyndham_analytics"

print("🔧 GENIE SPACE CONFIGURATION - NOTEBOOK 03")
print("=" * 60)
print(f"📊 Catalog: {catalog}")
print(f"📁 Schemas: {staging_schema}, {curated_schema}, {analytics_schema}")
print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Validate Data Foundation

# COMMAND ----------

# Validate that all required tables exist and have data
tables_to_validate = [
    f"{catalog}.{curated_schema}.properties_master",
    f"{catalog}.{curated_schema}.daily_performance", 
    f"{catalog}.{curated_schema}.competitive_intelligence",
    f"{catalog}.{curated_schema}.guest_transactions",
    f"{catalog}.{curated_schema}.market_events"
]

print("🔍 VALIDATING DATA FOUNDATION")
print("=" * 40)

validation_results = {}
for table in tables_to_validate:
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table}").collect()[0]['cnt']
        validation_results[table] = count
        status = "✅" if count > 0 else "❌"
        print(f"{status} {table}: {count:,} records")
    except Exception as e:
        validation_results[table] = f"ERROR: {str(e)}"
        print(f"❌ {table}: ERROR - {str(e)}")

print("\n📈 SUMMARY:")
total_records = sum([v for v in validation_results.values() if isinstance(v, int)])
print(f"Total Records: {total_records:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Data Quality Validation

# COMMAND ----------

# Validate key business metrics and data quality
print("🔍 DATA QUALITY VALIDATION")
print("=" * 40)

# Check RevPAR calculation accuracy
revpar_validation = spark.sql(f"""
SELECT 
    COUNT(*) as total_records,
    COUNT(CASE WHEN ABS(revpar - (adr * occupancy_rate)) > 0.01 THEN 1 END) as calculation_errors,
    MIN(occupancy_rate) as min_occupancy,
    MAX(occupancy_rate) as max_occupancy,
    AVG(occupancy_rate) as avg_occupancy,
    MIN(adr) as min_adr,
    MAX(adr) as max_adr,
    AVG(adr) as avg_adr,
    MIN(revpar) as min_revpar,
    MAX(revpar) as max_revpar,
    AVG(revpar) as avg_revpar
FROM {catalog}.{curated_schema}.daily_performance
""").collect()[0]

print("💰 REVPAR CALCULATION VALIDATION:")
print(f"   Total Records: {revpar_validation['total_records']:,}")
print(f"   Calculation Errors: {revpar_validation['calculation_errors']:,}")
print(f"   Occupancy Range: {revpar_validation['min_occupancy']:.1%} - {revpar_validation['max_occupancy']:.1%}")
print(f"   ADR Range: ${revpar_validation['min_adr']:.2f} - ${revpar_validation['max_adr']:.2f}")
print(f"   RevPAR Range: ${revpar_validation['min_revpar']:.2f} - ${revpar_validation['max_revpar']:.2f}")

# Validate brand distribution
brand_distribution = spark.sql(f"""
SELECT 
    brand,
    COUNT(*) as property_count
FROM {catalog}.{curated_schema}.properties_master
GROUP BY brand
ORDER BY brand
""").collect()

print("\n🏨 BRAND DISTRIBUTION:")
for row in brand_distribution:
    print(f"   {row['brand']}: {row['property_count']} properties")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Create Optimized Genie Views

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Create Comprehensive Revenue Analytics View

# COMMAND ----------

# Create the primary materialized view for Genie
print("🏗️ CREATING GENIE REVENUE ANALYTICS VIEW")
print("=" * 50)

# Drop existing view if it exists
spark.sql(f"DROP VIEW IF EXISTS {catalog}.{analytics_schema}.genie_revenue_analytics")

# Create comprehensive view with all necessary joins and calculations
genie_view_sql = f"""
CREATE VIEW {catalog}.{analytics_schema}.genie_revenue_analytics
COMMENT 'Comprehensive revenue analytics view optimized for Genie natural language queries. 
Combines property attributes, daily performance, and competitive intelligence with calculated business metrics.'
AS
SELECT 
    -- Property Identifiers and Attributes
    p.property_id,
    p.property_name,
    p.brand,
    p.region,
    p.market_tier,
    p.property_type,
    p.ownership_type,
    p.room_count,
    p.city,
    p.state_province,
    p.country,
    p.market_id,
    
    -- Date and Time Dimensions
    dp.business_date,
    YEAR(dp.business_date) as year,
    MONTH(dp.business_date) as month,
    DAYOFWEEK(dp.business_date) as day_of_week,
    DATE_FORMAT(dp.business_date, 'EEEE') as day_name,
    DATE_FORMAT(dp.business_date, 'MMMM') as month_name,
    QUARTER(dp.business_date) as quarter,
    
    -- Seasonality Classification
    CASE 
        WHEN MONTH(dp.business_date) IN (6,7,8) THEN 'Peak'
        WHEN MONTH(dp.business_date) IN (12,1,2) AND 
             DAY(dp.business_date) NOT BETWEEN 20 AND 31 THEN 'Low'
        WHEN MONTH(dp.business_date) = 12 AND DAY(dp.business_date) >= 20 THEN 'Peak'
        WHEN MONTH(dp.business_date) = 1 AND DAY(dp.business_date) <= 7 THEN 'Peak'
        ELSE 'Shoulder'
    END as season,
    
    -- Core Performance Metrics
    dp.rooms_available,
    dp.rooms_sold,
    dp.occupancy_rate,
    dp.adr,
    dp.revpar,
    dp.revenue_rooms,
    dp.revenue_total,
    
    -- Operational Metrics
    dp.avg_length_of_stay,
    dp.walk_in_rate,
    dp.no_show_rate,
    dp.cancellation_rate,
    
    -- Competitive Intelligence
    ci.market_occupancy,
    ci.market_adr,
    ci.market_revpar,
    ci.penetration_index,
    ci.adr_index,
    ci.revpar_index,
    
    -- Performance Categorization
    CASE 
        WHEN ci.revpar_index >= 110 THEN 'Strong Outperformer'
        WHEN ci.revpar_index >= 105 THEN 'Outperformer'
        WHEN ci.revpar_index >= 95 THEN 'In-line'
        WHEN ci.revpar_index >= 90 THEN 'Slight Underperformer'
        ELSE 'Underperformer'
    END as market_performance_category,
    
    -- Calculated Business Metrics
    dp.revenue_total / p.room_count as revenue_per_room,
    dp.revenue_rooms / dp.rooms_sold as revenue_per_sold_room,
    (dp.occupancy_rate * ci.market_occupancy) as demand_strength_indicator,
    
    -- Year-over-Year Comparison Helpers
    LAG(dp.revpar, 365) OVER (
        PARTITION BY dp.property_id 
        ORDER BY dp.business_date
    ) as revpar_last_year,
    
    LAG(dp.occupancy_rate, 365) OVER (
        PARTITION BY dp.property_id 
        ORDER BY dp.business_date
    ) as occupancy_last_year,
    
    LAG(dp.adr, 365) OVER (
        PARTITION BY dp.property_id 
        ORDER BY dp.business_date
    ) as adr_last_year

FROM {catalog}.{curated_schema}.daily_performance dp
JOIN {catalog}.{curated_schema}.properties_master p 
    ON dp.property_id = p.property_id
LEFT JOIN {catalog}.{curated_schema}.competitive_intelligence ci 
    ON dp.property_id = ci.property_id 
    AND dp.business_date = ci.business_date
"""

# Execute the view creation
spark.sql(genie_view_sql)
print("✅ Created genie_revenue_analytics view")

# Validate the view
view_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {catalog}.{analytics_schema}.genie_revenue_analytics").collect()[0]['cnt']
print(f"📊 View contains: {view_count:,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Create Property Lookup View

# COMMAND ----------

# Create simplified property lookup for property-focused queries
print("🏗️ CREATING PROPERTY LOOKUP VIEW")
print("=" * 40)

spark.sql(f"DROP VIEW IF EXISTS {catalog}.{analytics_schema}.genie_property_lookup")

property_lookup_sql = f"""
CREATE VIEW {catalog}.{analytics_schema}.genie_property_lookup
COMMENT 'Simplified property lookup with current performance metrics for property-focused queries.'
AS
SELECT DISTINCT
    p.property_id,
    p.property_name,
    p.brand,
    p.region,
    p.market_tier,
    p.property_type,
    p.city,
    p.state_province,
    p.room_count,
    p.ownership_type,
    
    -- Latest 30-day averages
    AVG(dp.revpar) OVER (
        PARTITION BY p.property_id 
        ORDER BY dp.business_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as avg_revpar_30d,
    
    AVG(dp.occupancy_rate) OVER (
        PARTITION BY p.property_id 
        ORDER BY dp.business_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as avg_occupancy_30d,
    
    AVG(dp.adr) OVER (
        PARTITION BY p.property_id 
        ORDER BY dp.business_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as avg_adr_30d

FROM {catalog}.{curated_schema}.properties_master p
JOIN {catalog}.{curated_schema}.daily_performance dp 
    ON p.property_id = dp.property_id
WHERE dp.business_date >= CURRENT_DATE - INTERVAL 30 DAYS
"""

spark.sql(property_lookup_sql)
print("✅ Created genie_property_lookup view")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: Genie Space Instructions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 General Instructions Template

# COMMAND ----------

# Define comprehensive Genie instructions
general_instructions = """
WYNDHAM REVENUE MANAGEMENT INTELLIGENCE SYSTEM

BUSINESS CONTEXT:
You are an AI assistant specialized in revenue management and pricing intelligence for Wyndham Hotels & Resorts. Your primary role is to help revenue managers, pricing analysts, and executives analyze property performance, competitive positioning, and market trends.

CORE BUSINESS DEFINITIONS:
• RevPAR (Revenue Per Available Room) = ADR × Occupancy Rate
  - This is the PRIMARY KPI for revenue performance analysis
  - Always use RevPAR as the default metric for "performance" unless specified otherwise
  - Example: If RevPAR = $85.50, this means each available room generated $85.50 in revenue

• ADR (Average Daily Rate) = Room Revenue ÷ Rooms Sold  
  - Represents the average price paid per occupied room
  - Higher ADR indicates premium pricing or market positioning

• Occupancy Rate = Rooms Sold ÷ Rooms Available
  - Expressed as decimal in data (0.75 = 75% occupancy)
  - Maximum realistic occupancy is typically 95% due to operational constraints

• Market Performance Index = (Property Metric ÷ Market Metric) × 100
  - Values >100 indicate outperformance vs market
  - Values <100 indicate underperformance vs market
  - RevPAR Index is most important for overall market performance

WYNDHAM BRAND HIERARCHY (Economy to Upscale):
1. Super 8, Travelodge (Economy segment)
2. Days Inn, Howard Johnson (Midscale Economy) 
3. Baymont, Ramada (Midscale)
4. Wingate (Upper Midscale)
5. Wyndham (Upscale)

SEASONALITY PATTERNS:
• Peak Season: June-August (summer travel), major holidays (Thanksgiving, Christmas/New Year)
• Shoulder Season: March-May (spring), September-November (fall)  
• Low Season: January-February, December (non-holiday periods)
• Weekly Patterns: Higher demand Thursday-Saturday, lower Sunday-Wednesday

GEOGRAPHIC REGIONS:
• Northeast: CT, MA, ME, NH, NJ, NY, PA, RI, VT
• Southeast: AL, FL, GA, KY, MS, NC, SC, TN, VA, WV
• Midwest: IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI
• Southwest: AZ, NM, NV, TX, UT
• West: AK, CA, CO, HI, ID, MT, OR, WA, WY
• Central Canada: Manitoba, Saskatchewan, Ontario
• Eastern Canada: Quebec, New Brunswick, Nova Scotia, Prince Edward Island
• Western Canada: British Columbia, Alberta

MARKET PERFORMANCE INTERPRETATION:
• RevPAR Index: >110 = Strong Outperformer, 105-110 = Outperformer, 95-105 = In-line, 90-95 = Slight Underperformer, <90 = Underperformer
• Penetration Index: Measures market share relative to fair share (room count based)
• Use competitive indices to distinguish market conditions from property-specific issues

QUERY RESPONSE GUIDELINES:
• Default to RevPAR for performance analysis unless user specifies ADR or occupancy
• Include year-over-year comparisons when analyzing trends or performance changes
• Consider seasonality when making performance assessments (don't compare peak to low season)
• For "best/worst performing" queries, rank by RevPAR unless context suggests otherwise
• When showing monetary values, format as currency (e.g., $85.50 not 85.5)
• When showing percentages, use % symbol (e.g., 75.2% not 0.752)
• Group regional analysis by the region field, not individual cities
• Always provide context for performance metrics (market conditions, seasonality, etc.)

TABLE RELATIONSHIPS:
• Primary table: genie_revenue_analytics (contains all necessary joins)
• Property details: Use property_name, brand, region fields for identification
• Date filtering: Use business_date field for date ranges
• Performance metrics: revpar, adr, occupancy_rate are core KPIs
• Market comparison: Use revpar_index, penetration_index, adr_index for competitive analysis
"""

print("📝 GENERAL INSTRUCTIONS DEFINED")
print("=" * 40)
print(f"Instruction length: {len(general_instructions):,} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Sample SQL Patterns

# COMMAND ----------

# Define sample SQL patterns for common queries
sample_sql_patterns = """
COMMON QUERY PATTERNS:

1. BASIC PERFORMANCE QUERY
Question: "What was our RevPAR last month?"
SQL Pattern:
SELECT AVG(revpar) as avg_revpar
FROM genie_revenue_analytics 
WHERE business_date >= DATE_SUB(DATE_TRUNC('MONTH', CURRENT_DATE), 1) 
  AND business_date < DATE_TRUNC('MONTH', CURRENT_DATE)

2. TOP PERFORMERS BY METRIC
Question: "Show me the top 10 properties by RevPAR this year"
SQL Pattern:
SELECT property_name, brand, region, AVG(revpar) as avg_revpar
FROM genie_revenue_analytics 
WHERE year = YEAR(CURRENT_DATE)
GROUP BY property_name, brand, region
ORDER BY avg_revpar DESC 
LIMIT 10

3. YEAR-OVER-YEAR COMPARISON
Question: "How did our Q3 RevPAR compare to last year?"
SQL Pattern:
SELECT 
  AVG(CASE WHEN year = YEAR(CURRENT_DATE) AND quarter = 3 THEN revpar END) as q3_current,
  AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 AND quarter = 3 THEN revpar END) as q3_last_year,
  ((AVG(CASE WHEN year = YEAR(CURRENT_DATE) AND quarter = 3 THEN revpar END) / 
    AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 AND quarter = 3 THEN revpar END)) - 1) * 100 as growth_pct
FROM genie_revenue_analytics
WHERE quarter = 3 AND year IN (YEAR(CURRENT_DATE)-1, YEAR(CURRENT_DATE))

4. MARKET PERFORMANCE ANALYSIS  
Question: "Which properties are outperforming their market?"
SQL Pattern:
SELECT property_name, brand, region, 
       AVG(revpar_index) as avg_revpar_index,
       AVG(penetration_index) as avg_penetration_index
FROM genie_revenue_analytics 
WHERE business_date >= CURRENT_DATE - INTERVAL 90 DAYS
GROUP BY property_name, brand, region
HAVING AVG(revpar_index) > 105
ORDER BY avg_revpar_index DESC

5. BRAND COMPARISON
Question: "Compare RevPAR performance by brand this year"
SQL Pattern:
SELECT brand, 
       AVG(revpar) as avg_revpar,
       AVG(occupancy_rate) as avg_occupancy,
       AVG(adr) as avg_adr,
       COUNT(*) as property_count
FROM genie_revenue_analytics
WHERE year = YEAR(CURRENT_DATE)
GROUP BY brand
ORDER BY avg_revpar DESC

6. SEASONAL ANALYSIS
Question: "How did summer performance compare across regions?"
SQL Pattern:
SELECT region,
       AVG(CASE WHEN season = 'Peak' THEN revpar END) as summer_revpar,
       AVG(CASE WHEN season = 'Peak' THEN occupancy_rate END) as summer_occupancy
FROM genie_revenue_analytics
WHERE season = 'Peak' AND year = YEAR(CURRENT_DATE)
GROUP BY region
ORDER BY summer_revpar DESC
"""

print("📝 SQL PATTERNS DEFINED")
print("=" * 30)
print(f"Pattern examples: {len(sample_sql_patterns.split('Question:')) - 1} patterns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4: Benchmark Question Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Define Benchmark Questions

# COMMAND ----------

# Define comprehensive benchmark questions with expected results
benchmark_questions = [
    {
        "tier": "basic",
        "question": "What was our average RevPAR last month?",
        "expected_logic": "Filter to previous month, calculate average RevPAR across all properties",
        "validation_sql": f"""
            SELECT ROUND(AVG(revpar), 2) as avg_revpar
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics 
            WHERE business_date >= DATE_SUB(DATE_TRUNC('MONTH', CURRENT_DATE), 1) 
              AND business_date < DATE_TRUNC('MONTH', CURRENT_DATE)
        """,
        "target_accuracy": 0.95
    },
    {
        "tier": "basic", 
        "question": "Which brand has the highest occupancy rate this year?",
        "expected_logic": "Group by brand, calculate average occupancy, return highest",
        "validation_sql": f"""
            SELECT brand, ROUND(AVG(occupancy_rate) * 100, 1) as avg_occupancy_pct
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics 
            WHERE year = YEAR(CURRENT_DATE)
            GROUP BY brand 
            ORDER BY avg_occupancy_pct DESC 
            LIMIT 1
        """,
        "target_accuracy": 0.95
    },
    {
        "tier": "basic",
        "question": "Show me the top 5 properties by RevPAR in the Northeast region",
        "expected_logic": "Filter Northeast, group by property, rank by RevPAR, limit 5",
        "validation_sql": f"""
            SELECT property_name, ROUND(AVG(revpar), 2) as avg_revpar
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics 
            WHERE region = 'Northeast'
            GROUP BY property_name 
            ORDER BY avg_revpar DESC 
            LIMIT 5
        """,
        "target_accuracy": 0.95
    },
    {
        "tier": "comparative",
        "question": "How did our RevPAR this quarter compare to the same quarter last year?",
        "expected_logic": "Compare current Q vs same Q last year, calculate growth percentage",
        "validation_sql": f"""
            SELECT 
                ROUND(AVG(CASE WHEN year = YEAR(CURRENT_DATE) AND quarter = QUARTER(CURRENT_DATE) THEN revpar END), 2) as current_quarter_revpar,
                ROUND(AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 AND quarter = QUARTER(CURRENT_DATE) THEN revpar END), 2) as last_year_revpar,
                ROUND(((AVG(CASE WHEN year = YEAR(CURRENT_DATE) AND quarter = QUARTER(CURRENT_DATE) THEN revpar END) / 
                        AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 AND quarter = QUARTER(CURRENT_DATE) THEN revpar END)) - 1) * 100, 1) as growth_percentage
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics
            WHERE quarter = QUARTER(CURRENT_DATE) AND year IN (YEAR(CURRENT_DATE)-1, YEAR(CURRENT_DATE))
        """,
        "target_accuracy": 0.90
    },
    {
        "tier": "comparative",
        "question": "Which properties are outperforming their market in terms of RevPAR?",
        "expected_logic": "Filter properties with RevPAR index > 100, order by performance",
        "validation_sql": f"""
            SELECT property_name, brand, region, ROUND(AVG(revpar_index), 1) as avg_revpar_index
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics 
            WHERE business_date >= CURRENT_DATE - INTERVAL 90 DAYS
              AND revpar_index IS NOT NULL
            GROUP BY property_name, brand, region
            HAVING AVG(revpar_index) > 100
            ORDER BY avg_revpar_index DESC
            LIMIT 10
        """,
        "target_accuracy": 0.90
    },
    {
        "tier": "comparative", 
        "question": "Compare weekend vs weekday performance by brand",
        "expected_logic": "Segment by day of week, compare Thu-Sat vs Sun-Wed performance",
        "validation_sql": f"""
            SELECT brand,
                ROUND(AVG(CASE WHEN day_of_week IN (1,5,6,7) THEN revpar END), 2) as weekend_revpar,
                ROUND(AVG(CASE WHEN day_of_week IN (2,3,4) THEN revpar END), 2) as weekday_revpar,
                ROUND(AVG(CASE WHEN day_of_week IN (1,5,6,7) THEN revpar END) - 
                      AVG(CASE WHEN day_of_week IN (2,3,4) THEN revpar END), 2) as weekend_premium
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics
            WHERE year = YEAR(CURRENT_DATE)
            GROUP BY brand
            ORDER BY weekend_premium DESC
        """,
        "target_accuracy": 0.90
    },
    {
        "tier": "complex",
        "question": "Identify our most seasonally sensitive properties based on RevPAR variance",
        "expected_logic": "Calculate standard deviation of monthly RevPAR by property, rank by volatility",
        "validation_sql": f"""
            SELECT property_name, brand, region,
                ROUND(AVG(revpar), 2) as avg_revpar,
                ROUND(STDDEV(revpar), 2) as revpar_std_dev,
                ROUND(STDDEV(revpar) / AVG(revpar) * 100, 1) as coefficient_of_variation
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics
            WHERE year = YEAR(CURRENT_DATE)
            GROUP BY property_name, brand, region
            HAVING COUNT(*) >= 90  -- Ensure sufficient data points
            ORDER BY coefficient_of_variation DESC
            LIMIT 10
        """,
        "target_accuracy": 0.85
    },
    {
        "tier": "complex",
        "question": "Which properties improved their market share penetration this year compared to last year?",
        "expected_logic": "Compare average penetration index YoY, filter for positive improvements",
        "validation_sql": f"""
            SELECT property_name, brand, region,
                ROUND(AVG(CASE WHEN year = YEAR(CURRENT_DATE) THEN penetration_index END), 1) as current_year_penetration,
                ROUND(AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 THEN penetration_index END), 1) as last_year_penetration,
                ROUND(AVG(CASE WHEN year = YEAR(CURRENT_DATE) THEN penetration_index END) - 
                      AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 THEN penetration_index END), 1) as penetration_improvement
            FROM {catalog}.{analytics_schema}.genie_revenue_analytics
            WHERE year IN (YEAR(CURRENT_DATE)-1, YEAR(CURRENT_DATE))
              AND penetration_index IS NOT NULL
            GROUP BY property_name, brand, region
            HAVING AVG(CASE WHEN year = YEAR(CURRENT_DATE) THEN penetration_index END) IS NOT NULL
               AND AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 THEN penetration_index END) IS NOT NULL
               AND AVG(CASE WHEN year = YEAR(CURRENT_DATE) THEN penetration_index END) > 
                   AVG(CASE WHEN year = YEAR(CURRENT_DATE)-1 THEN penetration_index END)
            ORDER BY penetration_improvement DESC
            LIMIT 15
        """,
        "target_accuracy": 0.85
    }
]

print("📋 BENCHMARK QUESTIONS DEFINED")
print("=" * 40)
print(f"Total questions: {len(benchmark_questions)}")
for tier in ["basic", "comparative", "complex"]:
    count = len([q for q in benchmark_questions if q["tier"] == tier])
    print(f"  {tier.title()} tier: {count} questions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Execute Benchmark Validation

# COMMAND ----------

# Execute benchmark questions to establish expected results
print("🧪 EXECUTING BENCHMARK VALIDATION")
print("=" * 50)

benchmark_results = []
for i, question in enumerate(benchmark_questions, 1):
    print(f"\n📝 Question {i} ({question['tier'].upper()}): {question['question']}")
    
    try:
        # Execute validation SQL
        result = spark.sql(question['validation_sql'])
        result_data = result.collect()
        
        # Store results
        question_result = {
            "question_id": i,
            "tier": question['tier'],
            "question": question['question'],
            "expected_logic": question['expected_logic'],
            "sql_executed": question['validation_sql'],
            "result_count": len(result_data),
            "result_data": [row.asDict() for row in result_data],
            "execution_status": "SUCCESS"
        }
        
        # Display first few results
        if len(result_data) > 0:
            print(f"✅ Executed successfully - {len(result_data)} results")
            if len(result_data) <= 5:
                for row in result_data:
                    print(f"   {dict(row.asDict())}")
            else:
                for row in result_data[:3]:
                    print(f"   {dict(row.asDict())}")
                print(f"   ... and {len(result_data)-3} more rows")
        