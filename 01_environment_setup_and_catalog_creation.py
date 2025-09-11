# Databricks notebook source
# MAGIC %md
# MAGIC # Wyndham Revenue Management Genie Space - Environment Setup
# MAGIC 
# MAGIC **Notebook 01**: Environment Setup and Catalog Creation  
# MAGIC **Runtime**: DBR 16.4 LTS  
# MAGIC **Data Scope**: ~900 properties (10% sample), US + Canada, 8 brands, 3 years historical  
# MAGIC 
# MAGIC This notebook sets up the foundational Unity Catalog structure for the Wyndham Revenue Management Genie space.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Validation

# COMMAND ----------

# Validate environment and get basic info
current_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]
print(f"Current User: {current_user}")
print(f"Spark Version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog Schema Creation

# COMMAND ----------

# Create schemas
schemas_to_create = [
    ('wyndham_staging', 'Raw data staging area for Wyndham revenue management data'),
    ('wyndham_curated', 'Clean, business-ready tables for Wyndham revenue analytics'),
    ('wyndham_analytics', 'Analytics views optimized for AI/BI Genie queries')
]

for schema_name, comment in schemas_to_create:
    try:
        spark.sql(f"""
            CREATE SCHEMA IF NOT EXISTS main.{schema_name}
            COMMENT '{comment}'
        """)
        print(f"✓ Schema: main.{schema_name}")
    except Exception as e:
        print(f"✗ Error creating schema main.{schema_name}: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Core Table Definitions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Properties Master Table
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.properties_master (
# MAGIC   property_id STRING NOT NULL COMMENT 'Unique property identifier. Format: WYN_[BRAND]_[REGION]_[NUMBER]. Example: WYN_DAYS_SE_001',
# MAGIC   property_name STRING NOT NULL COMMENT 'Full property name including brand and location. Example: Days Inn Atlanta Airport',
# MAGIC   brand STRING NOT NULL COMMENT 'Wyndham brand family. Valid values: "Days Inn", "Super 8", "Ramada", "Wyndham", "Baymont", "Travelodge", "Howard Johnson", "Wingate"',
# MAGIC   region STRING NOT NULL COMMENT 'Geographic region. Valid values: "Northeast", "Southeast", "Midwest", "Southwest", "West", "Central Canada", "Eastern Canada", "Western Canada"',
# MAGIC   market_tier STRING NOT NULL COMMENT 'Market classification. Valid values: "Primary" (major cities), "Secondary" (mid-size markets), "Tertiary" (small markets)',
# MAGIC   property_type STRING NOT NULL COMMENT 'Location type. Valid values: "Urban", "Suburban", "Airport", "Highway", "Resort", "Extended Stay"',
# MAGIC   room_count INT NOT NULL COMMENT 'Total number of available guest rooms. Typically 60-450 depending on brand',
# MAGIC   ownership_type STRING NOT NULL COMMENT 'Business model. Valid values: "Corporate", "Franchise", "Management Contract"',
# MAGIC   open_date DATE NOT NULL COMMENT 'Property opening date for age analysis',
# MAGIC   city STRING NOT NULL COMMENT 'Primary city location',
# MAGIC   state_province STRING NOT NULL COMMENT 'State (US) or province (Canada)',
# MAGIC   country STRING NOT NULL COMMENT 'Country location. Valid values: "US", "Canada"',
# MAGIC   market_id STRING NOT NULL COMMENT 'Market identifier for competitive analysis',
# MAGIC   latitude DECIMAL(9,6) COMMENT 'Geographic latitude for mapping',
# MAGIC   longitude DECIMAL(9,6) COMMENT 'Geographic longitude for mapping'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC COMMENT 'Master property data for all Wyndham properties. Foundation for property lookup, regional analysis, and brand performance comparisons.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Daily Performance Table
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.daily_performance (
# MAGIC   property_id STRING NOT NULL COMMENT 'Foreign key to properties_master.property_id',
# MAGIC   business_date DATE NOT NULL COMMENT 'Stay date for revenue recognition. Used for all time-based analysis',
# MAGIC   rooms_available INT NOT NULL COMMENT 'Total rooms available for sale excluding out-of-order rooms',
# MAGIC   rooms_sold INT NOT NULL COMMENT 'Actual rooms sold and occupied',
# MAGIC   occupancy_rate DECIMAL(5,4) NOT NULL COMMENT 'Occupancy percentage (rooms_sold/rooms_available). Values 0-1 (0.7500 = 75%)',
# MAGIC   adr DECIMAL(10,2) NOT NULL COMMENT 'Average Daily Rate in USD. Room revenue divided by rooms sold',
# MAGIC   revpar DECIMAL(10,2) NOT NULL COMMENT 'Revenue Per Available Room in USD. Primary performance KPI (adr * occupancy_rate)',
# MAGIC   revenue_rooms DECIMAL(12,2) NOT NULL COMMENT 'Total room revenue in USD',
# MAGIC   revenue_fb DECIMAL(12,2) COMMENT 'Food & Beverage revenue in USD',
# MAGIC   revenue_other DECIMAL(12,2) COMMENT 'Other revenue (parking, Wi-Fi, etc.) in USD',
# MAGIC   revenue_total DECIMAL(12,2) NOT NULL COMMENT 'Total property revenue in USD',
# MAGIC   avg_length_of_stay DECIMAL(3,1) NOT NULL COMMENT 'Average guest stay duration in nights',
# MAGIC   booking_channel_mix STRING COMMENT 'JSON distribution: {"direct": 0.45, "ota": 0.35, "gds": 0.20}',
# MAGIC   market_segment_mix STRING COMMENT 'JSON guest types: {"business": 0.60, "leisure": 0.30, "group": 0.10}',
# MAGIC   walk_in_rate DECIMAL(5,4) COMMENT 'Percentage of walk-in guests',
# MAGIC   no_show_rate DECIMAL(5,4) COMMENT 'Percentage of no-shows',
# MAGIC   cancellation_rate DECIMAL(5,4) COMMENT 'Same-day cancellation percentage'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (business_date)
# MAGIC COMMENT 'Daily operational performance metrics. Primary table for revenue analysis and KPI reporting.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Competitive Intelligence Table
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.competitive_intelligence (
# MAGIC   market_id STRING NOT NULL COMMENT 'Market identifier for competitive properties',
# MAGIC   business_date DATE NOT NULL COMMENT 'Performance measurement date',
# MAGIC   property_id STRING NOT NULL COMMENT 'Subject Wyndham property',
# MAGIC   market_occupancy DECIMAL(5,4) NOT NULL COMMENT 'Market average occupancy rate',
# MAGIC   market_adr DECIMAL(10,2) NOT NULL COMMENT 'Market average ADR in USD',
# MAGIC   market_revpar DECIMAL(10,2) NOT NULL COMMENT 'Market RevPAR in USD',
# MAGIC   penetration_index DECIMAL(5,2) NOT NULL COMMENT 'Market share performance. 100 = fair share, >100 = above fair share',
# MAGIC   adr_index DECIMAL(5,2) NOT NULL COMMENT 'ADR vs market. 100 = market average, >100 = premium',
# MAGIC   revpar_index DECIMAL(5,2) NOT NULL COMMENT 'RevPAR vs market. 100 = market average, primary metric',
# MAGIC   market_room_nights INT NOT NULL COMMENT 'Total market room nights demanded',
# MAGIC   fair_share_rooms INT NOT NULL COMMENT 'Expected room nights based on inventory share'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (business_date)
# MAGIC COMMENT 'Competitive performance for market share analysis. Critical for understanding market vs property performance.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Guest Transaction Table
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.guest_transactions (
# MAGIC   transaction_id STRING NOT NULL COMMENT 'Unique transaction identifier',
# MAGIC   property_id STRING NOT NULL COMMENT 'Foreign key to properties_master.property_id',
# MAGIC   guest_id STRING NOT NULL COMMENT 'Unique guest identifier for loyalty analysis',
# MAGIC   business_date DATE NOT NULL COMMENT 'Arrival date',
# MAGIC   departure_date DATE NOT NULL COMMENT 'Departure date',
# MAGIC   length_of_stay INT NOT NULL COMMENT 'Number of nights',
# MAGIC   room_type STRING NOT NULL COMMENT 'Room category. Examples: "Standard King", "Suite"',
# MAGIC   rate_code STRING NOT NULL COMMENT 'Rate plan. Examples: "BAR", "AAA", "CORP"',
# MAGIC   room_revenue DECIMAL(10,2) NOT NULL COMMENT 'Room revenue for entire stay in USD',
# MAGIC   total_revenue DECIMAL(10,2) NOT NULL COMMENT 'Total guest revenue in USD',
# MAGIC   booking_channel STRING NOT NULL COMMENT 'Distribution channel. Examples: "Direct", "Expedia", "Walk-in"',
# MAGIC   market_segment STRING NOT NULL COMMENT 'Guest type. Valid values: "Business", "Leisure", "Group", "Extended Stay"',
# MAGIC   booking_date DATE NOT NULL COMMENT 'Reservation creation date',
# MAGIC   advance_booking_days INT NOT NULL COMMENT 'Days between booking and arrival',
# MAGIC   guest_type STRING COMMENT 'Classification. Examples: "New", "Repeat", "Loyalty Member"',
# MAGIC   cancellation_date DATE COMMENT 'Cancellation date if applicable',
# MAGIC   no_show BOOLEAN COMMENT 'True if guest failed to arrive'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (business_date)
# MAGIC COMMENT 'Individual guest transactions for detailed revenue analysis and guest behavior patterns.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Market Events Table
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.market_events (
# MAGIC   event_id STRING NOT NULL COMMENT 'Unique event identifier',
# MAGIC   market_id STRING NOT NULL COMMENT 'Market where event occurs',
# MAGIC   event_date DATE NOT NULL COMMENT 'Event start date',
# MAGIC   end_date DATE COMMENT 'Event end date for multi-day events',
# MAGIC   event_name STRING NOT NULL COMMENT 'Event name or description',
# MAGIC   event_type STRING NOT NULL COMMENT 'Event category. Valid values: "Conference", "Sports", "Concert", "Holiday", "Weather", "Economic"',
# MAGIC   impact_rating STRING NOT NULL COMMENT 'Expected impact. Valid values: "Low", "Medium", "High"',
# MAGIC   demand_lift_pct DECIMAL(5,2) COMMENT 'Percentage demand increase',
# MAGIC   adr_lift_pct DECIMAL(5,2) COMMENT 'Rate premium opportunity percentage'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (event_date)
# MAGIC COMMENT 'External events affecting hotel demand for understanding performance anomalies.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Analytics Views

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Primary Revenue Analytics View - Pre-joined for Genie
# MAGIC CREATE OR REPLACE VIEW main.wyndham_analytics.revenue_performance_view AS
# MAGIC SELECT 
# MAGIC   -- Property Attributes
# MAGIC   p.property_id,
# MAGIC   p.property_name,
# MAGIC   p.brand,
# MAGIC   p.region,
# MAGIC   p.market_tier,
# MAGIC   p.property_type,
# MAGIC   p.ownership_type,
# MAGIC   p.room_count,
# MAGIC   p.city,
# MAGIC   p.state_province,
# MAGIC   p.country,
# MAGIC   
# MAGIC   -- Performance Metrics
# MAGIC   dp.business_date,
# MAGIC   dp.occupancy_rate,
# MAGIC   dp.adr,
# MAGIC   dp.revpar,
# MAGIC   dp.revenue_total,
# MAGIC   dp.avg_length_of_stay,
# MAGIC   
# MAGIC   -- Time Dimensions
# MAGIC   YEAR(dp.business_date) as year,
# MAGIC   MONTH(dp.business_date) as month_number,
# MAGIC   DATE_FORMAT(dp.business_date, 'MMMM') as month_name,
# MAGIC   QUARTER(dp.business_date) as quarter,
# MAGIC   CASE 
# MAGIC     WHEN DAYOFWEEK(dp.business_date) IN (1, 2, 3, 4) THEN 'Weekday'
# MAGIC     ELSE 'Weekend'
# MAGIC   END as weekend_indicator,
# MAGIC   
# MAGIC   -- Business Classifications
# MAGIC   CASE 
# MAGIC     WHEN MONTH(dp.business_date) IN (6, 7, 8) THEN 'Peak'
# MAGIC     WHEN MONTH(dp.business_date) IN (12, 1, 2) THEN 'Low' 
# MAGIC     ELSE 'Shoulder'
# MAGIC   END as season,
# MAGIC   
# MAGIC   -- Competitive Performance
# MAGIC   ci.market_revpar,
# MAGIC   ci.revpar_index,
# MAGIC   ci.penetration_index,
# MAGIC   CASE 
# MAGIC     WHEN ci.revpar_index >= 110 THEN 'Outperforming'
# MAGIC     WHEN ci.revpar_index >= 90 THEN 'In-line'
# MAGIC     ELSE 'Underperforming'
# MAGIC   END as market_performance_category,
# MAGIC   
# MAGIC   -- Calculated Metrics
# MAGIC   dp.revenue_total / p.room_count as revenue_per_room,
# MAGIC   LAG(dp.revpar, 365) OVER (PARTITION BY dp.property_id ORDER BY dp.business_date) as revpar_ly
# MAGIC   
# MAGIC FROM main.wyndham_curated.daily_performance dp
# MAGIC JOIN main.wyndham_curated.properties_master p 
# MAGIC   ON dp.property_id = p.property_id
# MAGIC LEFT JOIN main.wyndham_curated.competitive_intelligence ci 
# MAGIC   ON dp.property_id = ci.property_id 
# MAGIC   AND dp.business_date = ci.business_date;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validation and Configuration

# COMMAND ----------

# Validate setup
def validate_setup():
    print("=== SCHEMA VALIDATION ===")
    schemas = spark.sql("SHOW SCHEMAS IN main").collect()
    wyndham_schemas = [row.databaseName for row in schemas if 'wyndham' in row.databaseName]
    
    expected_schemas = ['wyndham_staging', 'wyndham_curated', 'wyndham_analytics']
    for schema in expected_schemas:
        status = "✓" if schema in wyndham_schemas else "✗"
        print(f"{status} Schema: main.{schema}")
    
    print("\n=== TABLE VALIDATION ===")
    tables = spark.sql("SHOW TABLES IN main.wyndham_curated").collect()
    table_names = [row.tableName for row in tables]
    
    expected_tables = ['properties_master', 'daily_performance', 'competitive_intelligence', 'guest_transactions', 'market_events']
    
    for table in expected_tables:
        status = "✓" if table in table_names else "✗"
        print(f"{status} Table: main.wyndham_curated.{table}")

validate_setup()

# COMMAND ----------

# Create configuration for data generation
import json

data_config = {
    "environment": {
        "catalog": "main",
        "curated_schema": "wyndham_curated",
        "analytics_schema": "wyndham_analytics"
    },
    "data_scope": {
        "property_count": 900,
        "brands": ["Days Inn", "Super 8", "Ramada", "Wyndham", "Baymont", "Travelodge", "Howard Johnson", "Wingate"],
        "regions": ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Central Canada", "Eastern Canada", "Western Canada"],
        "countries": ["US", "Canada"],
        "start_date": "2021-01-01",
        "end_date": "2023-12-31"
    }
}

# Store for next notebook
dbutils.fs.put("/tmp/wyndham_config.json", json.dumps(data_config, indent=2))
print("✓ Configuration saved for data generation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC **Environment Setup Complete**
# MAGIC 
# MAGIC - 3 Unity Catalog schemas created
# MAGIC - 5 core tables with comprehensive metadata
# MAGIC - 1 analytics view optimized for Genie
# MAGIC - Configuration saved for data generation
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC - Notebook 02: Generate realistic synthetic data
# MAGIC - Notebook 03: Genie space configuration
# MAGIC - Notebook 04: Testing and validation